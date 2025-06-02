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
        
        # Get the absolute path to the data file
        data_path = Path(__file__).parent.parent / self.config['output_dir'] / 'examples.json'
        print(f"Looking for data at: {data_path}")
        
        # Try alternative paths if the first one doesn't exist
        if not data_path.exists():
            alt_paths = [
                Path('/content/drive/MyDrive/CBAI/data/examples.json'),
                Path('/content/drive/MyDrive/CBAI/CBAI/data/examples.json'),
                Path('data/examples.json'),
                Path('../data/examples.json')
            ]
            for path in alt_paths:
                print(f"Trying alternative path: {path}")
                if path.exists():
                    print(f"Found data at: {path}")
                    data_path = path
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find examples.json. Tried paths:\n" +
                    f"1. {data_path}\n" +
                    "\n".join(f"{i+2}. {path}" for i, path in enumerate(alt_paths))
                )
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize models and tokenizers
        self.models = {}
        self.tokenizers = {}
        
        # Validate Hugging Face login
        try:
            user_info = whoami()
            print(f"\nAuthenticated as: {user_info['name']}")
        except Exception as e:
            print("\nNot logged in to Hugging Face. Please run:")
            print("huggingface-cli login")
            raise ValueError("Not logged in to Hugging Face")
        
        for model_name in self.config['benchmarking']['models']:
            print(f"\nLoading model: {model_name}")
            try:
                # Common model loading parameters with optimizations
                model_kwargs = {
                    'torch_dtype': torch.float16 if self.config['benchmarking']['use_fp16'] else torch.float32,
                    'device_map': "auto",
                    'trust_remote_code': True,
                    'use_safetensors': True,
                    'low_cpu_mem_usage': True,
                    'offload_folder': "offload",  # Offload weights to disk if needed
                    'offload_state_dict': True,  # Offload state dict to CPU
                    'max_memory': {0: "8GiB"},  # Limit GPU memory usage
                }
                
                # Add token if model requires authentication
                if any(restricted in model_name.lower() for restricted in ['llama', 'gpt', 'mistral', 'qwen']):
                    model_kwargs['token'] = True  # Use logged-in token
                
                # Load model with appropriate settings
                self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                
                # Common tokenizer parameters
                tokenizer_kwargs = {
                    'trust_remote_code': True,
                    'use_fast': True,
                    'model_max_length': 2048,  # Limit context length for faster processing
                }
                
                # Add token if model requires authentication
                if any(restricted in model_name.lower() for restricted in ['llama', 'gpt', 'mistral', 'qwen']):
                    tokenizer_kwargs['token'] = True  # Use logged-in token
                
                # Load tokenizer with appropriate settings
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name,
                    **tokenizer_kwargs
                )
                
                # Set up padding token if not present
                if self.tokenizers[model_name].pad_token is None:
                    if self.tokenizers[model_name].eos_token is not None:
                        self.tokenizers[model_name].pad_token = self.tokenizers[model_name].eos_token
                    else:
                        self.tokenizers[model_name].pad_token = '[PAD]'
                        self.tokenizers[model_name].add_special_tokens({'pad_token': '[PAD]'})
                
                print(f"Successfully loaded {model_name}")
            except ImportError as e:
                print(f"\nMissing required package: {str(e)}")
                print("Please install it using:")
                print("pip install 'accelerate>=0.26.0'")
                raise
            except Exception as e:
                print(f"\nError loading model {model_name}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("\nPossible solutions:")
                print("1. Check if you have access to the model on Hugging Face")
                print("2. Try running: huggingface-cli login")
                print("3. Check if you have enough disk space and memory")
                print("4. Check if the model name is correct and exists")
                print("5. Make sure all required packages are installed:")
                print("   pip install 'accelerate>=0.26.0'")
                raise
    
    def format_prompt(self, words: List[str], category: str) -> str:
        """Format the prompt for the model."""
        return f"""Given this list of words: {words}
How many words in this list belong to the category '{category}'?
Answer with a single number between 0 and {len(words)}.

Answer:"""

    def get_model_prediction(self, model_name: str, prompt: str) -> int:
        """Get prediction from the model."""
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,  # We only need a small number for the answer
                do_sample=False,   # No sampling for deterministic results
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,  # Force EOS after generation
                no_repeat_ngram_size=3,  # Prevent repetition
                bad_words_ids=[[tokenizer.encode("Answer:")[0]]]  # Prevent model from starting a new answer
            )
        
        # Decode and extract the number
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Debug logging for problematic responses
        if not any(c.isdigit() for c in response):
            print(f"\nWarning: No number found in response:")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
        
        # Extract just the number from the response
        try:
            # Find the last number in the response
            words = response.split()
            for word in reversed(words):
                # Remove any punctuation and try to convert to int
                clean_word = ''.join(c for c in word if c.isdigit())
                if clean_word:
                    prediction = int(clean_word)
                    return prediction
            return -1  # No number found
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Raw response: {response}")
            return -1  # Error in parsing

    def get_model_predictions_batch(self, model_name: str, prompts: List[str], batch_size: int = 8) -> List[int]:
        """Get predictions from model in batches."""
        predictions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Process each prompt individually to ensure correct model usage
            for prompt in batch_prompts:
                prediction = self.get_model_prediction(model_name, prompt)
                predictions.append(prediction)
            
            # Clear CUDA cache if using GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
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