import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import os
import sys
import locale
import urllib3
from urllib3.connection import HTTPConnection
from huggingface_hub import login, whoami

# Check for required packages
try:
    import accelerate
except ImportError:
    print("\nMissing required package: accelerate")
    print("Please install it using:")
    print("pip install 'accelerate>=0.26.0'")
    raise ImportError("accelerate package is required")

# Monkey patch urllib3 to handle Unicode headers
original_putheader = HTTPConnection.putheader
def patched_putheader(self, header, *values):
    # Convert Unicode to ASCII, replacing non-ASCII with closest ASCII equivalent
    def ascii_safe(value):
        if isinstance(value, str):
            return value.encode('ascii', 'replace').decode('ascii')
        return value
    
    values = [ascii_safe(v) for v in values]
    return original_putheader(self, header, *values)

HTTPConnection.putheader = patched_putheader

# Set multiple encoding environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# Set PyTorch optimizations
torch.backends.cudnn.benchmark = True  # Optimize CUDA operations
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matrix multiplication
torch.backends.cudnn.allow_tf32 = True  # Use TF32 for convolutions

class ModelBenchmarker:
    def __init__(self, config_path: str):
        # Get the absolute path to the config file
        config_path = Path(__file__).parent.parent / config_path
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        
        # Load test data
        data_path = Path(__file__).parent.parent / self.config['output_dir'] / self.config['data_file']
        
        # Try different possible paths for the data file
        possible_paths = [
            data_path,
            Path('/content/drive/MyDrive/CBAI/data') / self.config['data_file'],
            Path('/content/drive/MyDrive/CBAI/CBAI/data') / self.config['data_file'],
            Path('data') / self.config['data_file'],
            Path('../data') / self.config['data_file']
        ]
        
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        else:
            raise FileNotFoundError(
                f"Could not find {self.config['data_file']}. Tried paths:\n" +
                "\n".join(str(p) for p in possible_paths)
            )
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize models and tokenizers
        self.models = {}
        self.tokenizers = {}
        
        # Track loaded model configs to prevent duplicates
        loaded_model_configs = set()
        
        # Validate Hugging Face login
        try:
            user_info = whoami()
            print(f"\nAuthenticated as: {user_info['name']}")
        except Exception as e:
            print("\nNot logged in to Hugging Face. Please run:")
            print("huggingface-cli login")
            raise ValueError("Not logged in to Hugging Face")
        
        # Load each model specified in the config
        for model_name in self.config['benchmarking']['models']:
            print(f"\nLoading model: {model_name}")
            try:
                # Load tokenizer
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                
                # Determine if we need quantization based on model size
                needs_quantization = any(large_model in model_name.lower() for large_model in [
                    'mistral', 'llama', 'falcon', 'mpt', 'gpt-j', 'gpt-neox', 'opt-6.7b', 'opt-13b'
                ])
                
                if needs_quantization:
                    print(f"Using 4-bit quantization for large model: {model_name}")
                    self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        load_in_4bit=True,
                        device_map="auto"
                    )
                else:
                    # For smaller models, use regular loading
                    self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        device_map="auto"
                    )
                
                # Verify model uniqueness
                model_config = self.models[model_name].config.to_dict()
                # Create a simple string representation of key config parameters
                model_config_str = f"{model_name}_{model_config.get('model_type', '')}_{model_config.get('vocab_size', 0)}_{model_config.get('hidden_size', 0)}"
                if model_config_str in loaded_model_configs:
                    raise ValueError(f"Duplicate model configuration detected for {model_name}")
                loaded_model_configs.add(model_config_str)
                
                # Store model
                self.models[model_name] = self.models[model_name]
                
                # Set up padding token if not present
                if self.tokenizers[model_name].pad_token is None:
                    if self.tokenizers[model_name].eos_token is not None:
                        self.tokenizers[model_name].pad_token = self.tokenizers[model_name].eos_token
                    else:
                        self.tokenizers[model_name].pad_token = '[PAD]'
                        self.tokenizers[model_name].add_special_tokens({'pad_token': '[PAD]'})
                
                print(f"Successfully loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                raise
    
    def format_prompt(self, words: List[str], category: str) -> str:
        """Format the prompt for the model."""
        return f"""Task: Count how many words in a list belong to a specific category.

List: {words}
Category: {category}

Count the number of words that belong to the category '{category}'.
Your answer should be a single number.

Answer:"""

    def get_model_prediction(self, model_name: str, prompt: str) -> Tuple[int, Dict]:
        """Get prediction from the model and return both the prediction and raw response."""
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # Check model's context length
        max_length = getattr(model.config, 'max_position_embeddings', None)
        if max_length is None:
            # Try to get from model config
            max_length = getattr(model.config, 'n_positions', None)
        if max_length is None:
            # Default to a conservative value if we can't determine
            max_length = 2048
        
        # Tokenize input to check length
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs['input_ids'].shape[1]
        
        # Add some buffer for the response (e.g., 100 tokens)
        if input_length + 100 > max_length:
            print(f"Warning: Input length {input_length} + response buffer exceeds model's max context length {max_length}")
            return -1, {
                'model': model_name,
                'prompt': prompt,
                'raw_output': None,
                'input_tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
                'output_tokens': None,
                'input_shape': inputs['input_ids'].shape,
                'output_shape': None,
                'error': 'context_length_exceeded'
            }
        
        # Common generation parameters for all models
        gen_kwargs = {
            'max_new_tokens': 5,  # Reduced from 10 to 5 since we only need a number
            'num_return_sequences': 1,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'forced_eos_token_id': tokenizer.eos_token_id,
            'no_repeat_ngram_size': 3,
        }
        
        # Add temperature only for models that support it
        if model_name.startswith(('EleutherAI/pythia', 'facebook/opt', 'mistralai')):
            gen_kwargs['do_sample'] = True  # Need to enable sampling for temperature
            gen_kwargs['temperature'] = 0.1  # Set small for more deterministic outputs
        elif model_name.startswith('bigscience/bloom'):
            # BLOOM doesn't support temperature, use do_sample=False for deterministic outputs
            gen_kwargs['do_sample'] = False
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Create raw response object
        raw_response = {
            'model': model_name,
            'prompt': prompt,
            'raw_output': response,
            'input_tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'output_tokens': tokenizer.convert_ids_to_tokens(outputs[0]),
            'input_shape': inputs['input_ids'].shape,
            'output_shape': outputs.shape,
            'context_length': {
                'max': max_length,
                'input': input_length,
                'output': outputs.shape[1]
            }
        }
        
        # Extract number from response
        prediction = self._extract_number_from_response(response, prompt)
        raw_response['extracted_prediction'] = prediction
        
        return prediction, raw_response

    def _extract_number_from_response(self, response: str, prompt: str) -> int:
        """Extract a number from the model's response.
        
        The model's response is defined as everything after the prompt.
        We look for the first number in the response.
        """
        try:
            # Find where the prompt ends in the response
            prompt_words = prompt.split()
            response_words = response.split()
            
            # Find the prompt end index
            prompt_end_idx = -1
            for i in range(len(response_words) - len(prompt_words) + 1):
                if response_words[i:i+len(prompt_words)] == prompt_words:
                    prompt_end_idx = i + len(prompt_words)
                    break
            
            if prompt_end_idx == -1:
                return -1  # Couldn't find where the prompt ends
            
            # Look for the first number in the response (after the prompt)
            for word in response_words[prompt_end_idx:]:
                # Remove any punctuation and try to convert to int
                clean_word = ''.join(c for c in word if c.isdigit())
                if clean_word:
                    return int(clean_word)
            
            return -1  # No number found in the response
            
        except Exception as e:
            print(f"Error extracting number: {str(e)}")
            return -1

    def get_model_predictions_batch(self, model_name: str, prompts: List[str], batch_size: int = 8) -> List[int]:
        """Get predictions from model in batches."""
        predictions = []
        raw_responses = []  # Store all raw responses
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Process each prompt individually to ensure correct model usage
            for prompt in batch_prompts:
                prediction, raw_response = self.get_model_prediction(model_name, prompt)
                predictions.append(prediction)
                raw_responses.append(raw_response)
            
            # Clear CUDA cache if using GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Save raw responses to file
        output_dir = Path(self.config['output_dir'])
        raw_responses_file = output_dir / f'raw_responses_{model_name.replace("/", "_")}.json'
        with open(raw_responses_file, 'w') as f:
            json.dump(raw_responses, f, indent=2)
        
        return predictions
    
    def _calculate_metrics(self, examples: List[Dict], model_name: str) -> Dict:
        """Calculate metrics for a subset of examples."""
        if not examples:  # Handle case where there are no examples
            return {
                'accuracy': 0.0,
                'error_distribution': {},
                'category_performance': {},
                'list_length_performance': {},
                'target_count_performance': {},
                'response_distribution': {}  # Track distribution of model responses
            }
            
        results = {
            'accuracy': 0.0,
            'error_distribution': defaultdict(int),
            'category_performance': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'list_length_performance': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'target_count_performance': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'response_distribution': defaultdict(int)  # Track distribution of model responses
        }
        
        # Prepare all prompts
        prompts = [self.format_prompt(example['words'], example['category']) for example in examples]
        
        # Get predictions in batches
        predictions = self.get_model_predictions_batch(model_name, prompts, batch_size=1)
        
        # Process results
        for example, prediction in zip(examples, predictions):
            # Skip invalid predictions
            if prediction == -1:
                continue
                
            # Track response distribution
            results['response_distribution'][prediction] += 1
            
            # Update metrics
            is_correct = prediction == example['num_target']
            results['accuracy'] += 1 if is_correct else 0
            
            # Track error distribution
            if not is_correct:
                results['error_distribution'][abs(prediction - example['num_target'])] += 1
            
            # Track performance by category
            results['category_performance'][example['category']]['total'] += 1
            if is_correct:
                results['category_performance'][example['category']]['correct'] += 1
            
            # Track performance by list length
            list_length = len(example['words'])
            results['list_length_performance'][list_length]['total'] += 1
            if is_correct:
                results['list_length_performance'][list_length]['correct'] += 1
            
            # Track performance by target count
            results['target_count_performance'][example['num_target']]['total'] += 1
            if is_correct:
                results['target_count_performance'][example['num_target']]['correct'] += 1
        
        # Calculate final accuracy
        total_valid_predictions = sum(1 for p in predictions if p != -1)
        results['accuracy'] = results['accuracy'] / total_valid_predictions if total_valid_predictions > 0 else 0
        
        # Calculate category-wise accuracy
        for category in results['category_performance']:
            perf = results['category_performance'][category]
            perf['accuracy'] = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        
        # Calculate list length-wise accuracy
        for length in results['list_length_performance']:
            perf = results['list_length_performance'][length]
            perf['accuracy'] = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        
        # Calculate target count-wise accuracy
        for count in results['target_count_performance']:
            perf = results['target_count_performance'][count]
            perf['accuracy'] = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        
        # Calculate response distribution percentages
        total_responses = sum(results['response_distribution'].values())
        if total_responses > 0:
            results['response_distribution_percentages'] = {
                k: v/total_responses for k, v in results['response_distribution'].items()
            }
        
        return results
    
    def evaluate_model(self, model_name: str) -> Dict:
        """Evaluate a single model on the dataset."""
        # Split data by generation method
        systematic_examples = [ex for ex in self.data if ex['generation_method'] == 'systematic']
        llm_examples = [ex for ex in self.data if ex['generation_method'] == 'llm']
        
        # Calculate metrics for each subset
        systematic_results = self._calculate_metrics(systematic_examples, model_name)  # Use original model name
        llm_results = self._calculate_metrics(llm_examples, model_name)  # Use original model name
        
        # Calculate overall metrics
        overall_results = self._calculate_metrics(self.data, model_name)  # Use original model name
        
        return {
            'systematic': systematic_results,
            'llm': llm_results,
            'overall': overall_results,
            'counts': {
                'systematic': len(systematic_examples),
                'llm': len(llm_examples),
                'total': len(self.data)
            }
        }
    
    def run_benchmarks(self):
        """Run benchmarks for all models."""
        results = {}
        for model_name in self.config['benchmarking']['models']:
            print(f"\nBenchmarking {model_name}...")
            results[model_name] = self.evaluate_model(model_name)
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        output_file = output_dir / 'benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved benchmark results to {output_file}")
        
        # Print summary
        print("\nBenchmark Summary:")
        print("-" * 50)
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            
            # Print systematic results
            print("\nSystematic Examples:")
            print(f"Count: {result['counts']['systematic']}")
            print(f"Accuracy: {result['systematic']['accuracy']:.3f}")
            
            # Print response distribution
            print("\nResponse Distribution (Systematic):")
            for response, count in sorted(result['systematic']['response_distribution'].items()):
                percentage = result['systematic']['response_distribution_percentages'][response] * 100
                print(f"  {response}: {count} times ({percentage:.1f}%)")
            
            print("\nCategory-wise Performance (Systematic):")
            for category, perf in result['systematic']['category_performance'].items():
                print(f"  {category}: {perf['accuracy']:.3f} ({perf['correct']}/{perf['total']})")
            
            # Print LLM results
            print("\nLLM Examples:")
            print(f"Count: {result['counts']['llm']}")
            print(f"Accuracy: {result['llm']['accuracy']:.3f}")
            
            # Print response distribution for LLM examples
            print("\nResponse Distribution (LLM):")
            for response, count in sorted(result['llm']['response_distribution'].items()):
                percentage = result['llm']['response_distribution_percentages'][response] * 100
                print(f"  {response}: {count} times ({percentage:.1f}%)")
            
            print("\nCategory-wise Performance (LLM):")
            for category, perf in result['llm']['category_performance'].items():
                print(f"  {category}: {perf['accuracy']:.3f} ({perf['correct']}/{perf['total']})")
            
            # Print overall results
            print("\nOverall Performance:")
            print(f"Total Examples: {result['counts']['total']}")
            print(f"Accuracy: {result['overall']['accuracy']:.3f}")
            
            # Print overall response distribution
            print("\nResponse Distribution (Overall):")
            for response, count in sorted(result['overall']['response_distribution'].items()):
                percentage = result['overall']['response_distribution_percentages'][response] * 100
                print(f"  {response}: {count} times ({percentage:.1f}%)")
            
            print("\nList Length Performance:")
            for length, perf in sorted(result['overall']['list_length_performance'].items()):
                print(f"  Length {length}: {perf['accuracy']:.3f} ({perf['correct']}/{perf['total']})")
            
            print("\nTarget Count Performance:")
            for count, perf in sorted(result['overall']['target_count_performance'].items()):
                print(f"  Count {count}: {perf['accuracy']:.3f} ({perf['correct']}/{perf['total']})")

if __name__ == "__main__":
    benchmarker = ModelBenchmarker('config.yaml')
    benchmarker.run_benchmarks() 