import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random

class ActivationPatching:
    def __init__(self, config_path: str):
        config_path = Path(__file__).parent.parent / config_path
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model and tokenizer
        self.model_name = self.config['mediation']['model']
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Determine if we need quantization based on model size
        needs_quantization = any(large_model in self.model_name.lower() for large_model in [
            'mistral', 'llama', 'falcon', 'mpt', 'gpt-j', 'gpt-neox', 'opt-6.7b', 'opt-13b'
        ])
        
        if needs_quantization:
            print(f"Using 4-bit quantization for large model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                device_map="auto",
                output_hidden_states=True  # We need hidden states for analysis
            )
        else:
            # For smaller models, use regular loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                output_hidden_states=True  # We need hidden states for analysis
            )
        
        # Store hooks for activation patching
        self.hooks = []
        self.activations = {}
        
    def _get_activation_hook(self, layer_idx: int):
        """Create a hook to store activations for a specific layer."""
        def hook(module, input, output):
            self.activations[f"layer_{layer_idx}"] = output.detach()
        return hook
    
    def _get_model_layers(self):
        """Get the layers of the model, handling different architectures."""
        if hasattr(self.model, 'transformer'):
            # BLOOM, OPT style
            return self.model.transformer.h
        elif hasattr(self.model, 'model'):
            # Mistral, LLaMA style
            return self.model.model.layers
        else:
            raise ValueError(f"Unsupported model architecture: {self.model.__class__.__name__}")
    
    def register_hooks(self, layer_idx: int):
        """Register hooks for a specific layer."""
        # Clear previous hooks
        self.remove_hooks()
        
        # Get the layer we want to patch
        layers = self._get_model_layers()
        layer = layers[layer_idx]
        
        # Register hook for the layer's output
        hook = layer.register_forward_hook(self._get_activation_hook(layer_idx))
        self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def format_prompt(self, words: List[str], category: str) -> str:
        """Format the prompt for the model."""
        return f"""Task: Count how many words in a list belong to a specific category.

List: {words}
Category: {category}

Count the number of words that belong to the category '{category}'.
Your answer should be a single number.

Answer:"""
    
    def get_model_prediction(self, prompt: str) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Get prediction and activations from the model."""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Set up generation parameters based on model type
            gen_kwargs = {
                'max_new_tokens': 5,
                'num_return_sequences': 1,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'forced_eos_token_id': self.tokenizer.eos_token_id,
                'no_repeat_ngram_size': 3,
            }
            
            # Add temperature only for models that support it
            if self.model_name.startswith(('EleutherAI/pythia', 'facebook/opt', 'mistralai')):
                gen_kwargs['do_sample'] = True
                gen_kwargs['temperature'] = 0.1
            else:
                # BLOOM doesn't support temperature, use do_sample=False for deterministic outputs
                gen_kwargs['do_sample'] = False
            
            # Generate prediction
            gen_outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
            # Decode the response - handle the output format correctly
            if hasattr(gen_outputs, 'sequences'):  # If it's a GenerateDecoderOnlyOutput
                token_ids = gen_outputs.sequences
            elif isinstance(gen_outputs, torch.Tensor):
                token_ids = gen_outputs
            else:
                # If it's a list or other format, convert to tensor first
                token_ids = torch.tensor(gen_outputs)
            
            # Ensure we have a 2D tensor
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            
            # Decode the response
            response = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
            
            print(f"\nDebug - Prompt: {prompt}")
            print(f"Debug - Raw response: {response}")
            
            # Extract prediction using the same method as test file
            try:
                # Find where the prompt ends in the response
                prompt_words = prompt.split()
                response_words = response.split()
                
                print(f"Debug - Prompt words: {prompt_words}")
                print(f"Debug - Response words: {response_words}")
                
                # Find the prompt end index
                prompt_end_idx = -1
                for i in range(len(response_words) - len(prompt_words) + 1):
                    if response_words[i:i+len(prompt_words)] == prompt_words:
                        prompt_end_idx = i + len(prompt_words)
                        break
                
                print(f"Debug - Prompt end index: {prompt_end_idx}")
                
                if prompt_end_idx == -1:
                    print("Debug - Couldn't find where the prompt ends")
                    prediction = -1  # Couldn't find where the prompt ends
                else:
                    # Look for the first number in the response (after the prompt)
                    print(f"Debug - Looking for numbers in: {response_words[prompt_end_idx:]}")
                    for word in response_words[prompt_end_idx:]:
                        # Remove any punctuation and try to convert to int
                        clean_word = ''.join(c for c in word if c.isdigit())
                        if clean_word:
                            prediction = int(clean_word)
                            print(f"Debug - Found number: {prediction}")
                            break
                    else:
                        print("Debug - No number found in response")
                        prediction = -1  # No number found in the response
            except Exception as e:
                print(f"Debug - Error extracting number: {str(e)}")
                prediction = -1
        
        # Get activations for each layer
        activations = {
            f"layer_{i}": hidden_states.detach() 
            for i, hidden_states in enumerate(outputs.hidden_states)
        }
        
        return prediction, activations
    
    def patch_activations(self, 
                         source_activations: Dict[str, torch.Tensor],
                         target_activations: Dict[str, torch.Tensor],
                         layer_idx: int,
                         token_positions: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Patch activations from source to target for a specific layer."""
        patched_activations = target_activations.copy()
        
        if token_positions is None:
            # Patch entire layer
            patched_activations[f"layer_{layer_idx}"] = source_activations[f"layer_{layer_idx}"]
        else:
            # Patch specific token positions
            layer_activations = patched_activations[f"layer_{layer_idx}"].clone()
            for pos in token_positions:
                layer_activations[0, pos] = source_activations[f"layer_{layer_idx}"][0, pos]
            patched_activations[f"layer_{layer_idx}"] = layer_activations
        
        return patched_activations
    
    def run_mediation_analysis(self, 
                             list_a: List[str],
                             list_b: List[str],
                             category_a: str,
                             category_b: str,
                             layer_idx: int,
                             token_positions: Optional[List[int]] = None) -> Dict:
        """Run mediation analysis for a specific layer and token positions."""
        # Get original predictions and activations
        prompt_a = self.format_prompt(list_a, category_a)
        prompt_b = self.format_prompt(list_b, category_b)
        
        print(f"\nAnalyzing layer {layer_idx}")
        print(f"List A: {list_a} (Category: {category_a})")
        print(f"List B: {list_b} (Category: {category_b})")
        
        pred_a, acts_a = self.get_model_prediction(prompt_a)
        pred_b, acts_b = self.get_model_prediction(prompt_b)
        
        print(f"\nOriginal predictions:")
        print(f"List A prediction: {pred_a}")
        print(f"List B prediction: {pred_b}")
        
        # Calculate total effect (preserving direction)
        total_effect = pred_b - pred_a
        print(f"Total effect: {total_effect}")
        
        if total_effect == 0:
            print("Warning: Total effect is 0, skipping layer analysis")
            return {
                'total_effect': 0,
                'direct_effect': 0,
                'indirect_effect': 0,
                'proportion_mediated': 0,
                'original_predictions': {
                    'list_a': pred_a,
                    'list_b': pred_b,
                    'patched': None
                },
                'diagnostics': {
                    'error': 'zero_total_effect',
                    'prompt_a': prompt_a,
                    'prompt_b': prompt_b
                }
            }
        
        # Patch activations and get patched prediction
        patched_acts = self.patch_activations(acts_a, acts_b, layer_idx, token_positions)
        
        # Run model with patched activations
        inputs = self.tokenizer(prompt_b, return_tensors="pt").to(self.model.device)
        
        # Create a custom forward pass that uses patched activations
        def custom_forward(*args, **kwargs):
            if hasattr(self.model, 'transformer'):
                # BLOOM, OPT style
                outputs = self.model.transformer(*args, **kwargs)
                # Convert hidden states tuple to list for modification
                hidden_states = list(outputs.hidden_states)
                # Replace the layer's activations
                hidden_states[layer_idx] = patched_acts[f"layer_{layer_idx}"]
                # Convert back to tuple
                outputs.hidden_states = tuple(hidden_states)
                return outputs
            elif hasattr(self.model, 'model'):
                # Mistral, LLaMA style
                outputs = self.model.model(*args, **kwargs)
                # Convert hidden states tuple to list for modification
                hidden_states = list(outputs.hidden_states)
                # Replace the layer's activations
                hidden_states[layer_idx] = patched_acts[f"layer_{layer_idx}"]
                # Convert back to tuple
                outputs.hidden_states = tuple(hidden_states)
                return outputs
            else:
                raise ValueError(f"Unsupported model architecture: {self.model.__class__.__name__}")
        
        with torch.no_grad():
            # Run the model with our custom forward pass
            transformer_outputs = custom_forward(**inputs)
            # Get the final hidden state
            hidden_states = transformer_outputs.hidden_states[-1]
            # Get logits from the final layer
            logits = self.model.lm_head(hidden_states)
            # Get the most likely token IDs
            token_ids = logits.argmax(dim=-1)
            # Ensure token_ids is a tensor
            if isinstance(token_ids, list):
                token_ids = torch.tensor(token_ids)
            # Generate prediction
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                num_return_sequences=1,
                return_legacy_cache=True  # Force legacy cache format
            )
            
            # Handle outputs based on type
            if hasattr(outputs, 'sequences'):  # If it's a Cache object
                token_ids = outputs.sequences
            elif isinstance(outputs, (list, tuple)):
                token_ids = torch.tensor(outputs)
            else:
                token_ids = outputs
                
            # Ensure we have a 2D tensor
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
                
            # Convert to list for decoding
            token_ids_list = token_ids[0].tolist()
            patched_response = self.tokenizer.decode(token_ids_list, skip_special_tokens=True)
        
        try:
            patched_pred = int([word for word in patched_response.split() if word.isdigit()][-1])
        except:
            patched_pred = -1
        
        print(f"\nPatched prediction: {patched_pred}")
        print(f"Patched response: {patched_response}")
        
        # Calculate direct and indirect effects (preserving direction)
        direct_effect = patched_pred - pred_a
        indirect_effect = pred_b - patched_pred
        
        print(f"Direct effect: {direct_effect}")
        print(f"Indirect effect: {indirect_effect}")
        
        # Calculate proportion mediated
        proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0
        print(f"Proportion mediated: {proportion_mediated}")
        
        return {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'proportion_mediated': proportion_mediated,
            'original_predictions': {
                'list_a': pred_a,
                'list_b': pred_b,
                'patched': patched_pred
            },
            'diagnostics': {
                'prompt_a': prompt_a,
                'prompt_b': prompt_b,
                'patched_response': patched_response,
                'token_positions': token_positions
            }
        }
    
    def analyze_layer_importance(self, 
                               list_pairs: List[Tuple[List[str], List[str], str, str]],
                               token_positions: Optional[List[int]] = None) -> Dict:
        """Analyze the importance of each layer for running count representation."""
        results = {}
        diagnostics = {
            'zero_effect_pairs': [],
            'invalid_predictions': [],
            'layer_stats': defaultdict(lambda: {
                'total_pairs': 0,
                'valid_pairs': 0,
                'zero_effect_pairs': 0,
                'invalid_predictions': 0
            })
        }
        
        # Get number of layers
        num_layers = len(self._get_model_layers())
        print(f"\nAnalyzing {len(list_pairs)} pairs")
        print(f"Number of layers: {num_layers}")
        
        for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
            layer_results = []
            layer_diagnostics = diagnostics['layer_stats'][layer_idx]
            
            for pair_idx, (list_a, list_b, category_a, category_b) in enumerate(list_pairs):
                result = self.run_mediation_analysis(
                    list_a, list_b, category_a, category_b, layer_idx, token_positions
                )
                
                # Track statistics
                layer_diagnostics['total_pairs'] += 1
                
                # Check for zero total effect
                if result['total_effect'] == 0:
                    layer_diagnostics['zero_effect_pairs'] += 1
                    diagnostics['zero_effect_pairs'].append({
                        'layer': layer_idx,
                        'pair_idx': pair_idx,
                        'list_a': list_a,
                        'list_b': list_b,
                        'category_a': category_a,
                        'category_b': category_b,
                        'predictions': result['original_predictions']
                    })
                    continue
                
                # Check for invalid predictions
                if result['original_predictions']['patched'] == -1:
                    layer_diagnostics['invalid_predictions'] += 1
                    diagnostics['invalid_predictions'].append({
                        'layer': layer_idx,
                        'pair_idx': pair_idx,
                        'list_a': list_a,
                        'list_b': list_b,
                        'category_a': category_a,
                        'category_b': category_b,
                        'predictions': result['original_predictions']
                    })
                    continue
                
                layer_diagnostics['valid_pairs'] += 1
                layer_results.append(result)
            
            # Aggregate results for this layer
            if layer_results:
                results[layer_idx] = {
                    'mean_proportion_mediated': np.mean([r['proportion_mediated'] for r in layer_results]),
                    'std_proportion_mediated': np.std([r['proportion_mediated'] for r in layer_results]),
                    'mean_indirect_effect': np.mean([r['indirect_effect'] for r in layer_results]),
                    'std_indirect_effect': np.std([r['indirect_effect'] for r in layer_results]),
                    'individual_results': layer_results
                }
            else:
                results[layer_idx] = {
                    'mean_proportion_mediated': 0,
                    'std_proportion_mediated': 0,
                    'mean_indirect_effect': 0,
                    'std_indirect_effect': 0,
                    'individual_results': []
                }
        
        # Print diagnostic summary
        print("\nDiagnostic Summary:")
        print("-" * 50)
        print(f"Total pairs analyzed: {len(list_pairs)}")
        print(f"Pairs with zero total effect: {len(diagnostics['zero_effect_pairs'])}")
        print(f"Pairs with invalid predictions: {len(diagnostics['invalid_predictions'])}")
        print("\nLayer-wise statistics:")
        for layer_idx in range(num_layers):
            stats = diagnostics['layer_stats'][layer_idx]
            print(f"\nLayer {layer_idx}:")
            print(f"  Total pairs: {stats['total_pairs']}")
            print(f"  Valid pairs: {stats['valid_pairs']}")
            print(f"  Zero effect pairs: {stats['zero_effect_pairs']}")
            print(f"  Invalid predictions: {stats['invalid_predictions']}")
        
        # Add diagnostics to results
        results['diagnostics'] = diagnostics
        
        return results
    
    def plot_layer_importance(self, results: Dict, output_path: str):
        """Plot the importance of each layer for running count representation."""
        # Remove the 'diagnostics' key if it exists
        if 'diagnostics' in results:
            results = {k: v for k, v in results.items() if k != 'diagnostics'}
        
        layers = list(results.keys())
        proportions = []
        stds = []
        
        for layer in layers:
            if 'mean_proportion_mediated' in results[layer]:
                proportions.append(results[layer]['mean_proportion_mediated'])
                stds.append(results[layer]['std_proportion_mediated'])
            else:
                # If no valid results for this layer, use 0
                proportions.append(0)
                stds.append(0)
        
        plt.figure(figsize=(12, 6))
        plt.bar(layers, proportions, yerr=stds, capsize=5)
        plt.xlabel('Layer Index')
        plt.ylabel('Mean Proportion Mediated')
        plt.title('Layer Importance for Running Count Representation')
        
        # Add text showing number of valid pairs for each layer
        for i, layer in enumerate(layers):
            valid_pairs = len(results[layer].get('individual_results', []))
            plt.text(layer, proportions[i], f'n={valid_pairs}', 
                    ha='center', va='bottom')
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path)
        plt.close()

def generate_list_pairs() -> List[Tuple[List[str], List[str], str, str]]:
    """Generate random pairs of word lists from existing examples.
    
    Returns:
        List of tuples containing (list_a, list_b, category_a, category_b)
    """
    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get number of pairs from config
    num_pairs = config['mediation']['num_list_pairs']
    
    # Try different possible paths for the data file
    possible_paths = [
        Path(config['output_dir']) / config['data_file'],
        Path('/content/drive/MyDrive/CBAI/data') / config['data_file'],
        Path('/content/drive/MyDrive/CBAI/CBAI/data') / config['data_file'],
        Path('data') / config['data_file'],
        Path('../data') / config['data_file']
    ]
    
    # Find the data file
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            print(f"Found data file at: {data_path}")
            break
    
    if data_path is None:
        raise FileNotFoundError(
            f"Could not find {config['data_file']}. Tried paths:\n" +
            "\n".join(str(p) for p in possible_paths)
        )
    
    # Load examples from the data file
    with open(data_path, 'r') as f:
        examples = json.load(f)
    
    if not examples:
        raise ValueError("No examples found in the data file")
    
    print(f"\nFound {len(examples)} total examples")
    
    # Generate random pairs
    selected_pairs = []
    attempts = 0
    max_attempts = num_pairs * 10  # Limit attempts to avoid infinite loops
    
    while len(selected_pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        
        # Randomly select two examples
        ex1, ex2 = random.sample(examples, 2)
        
        # Add the pair if it has different target counts
        if ex1['num_target'] != ex2['num_target']:
            selected_pairs.append((ex1['words'], ex2['words'], ex1['category'], ex2['category']))
    
    if len(selected_pairs) < num_pairs:
        print(f"\nWarning: Only found {len(selected_pairs)} valid pairs after {attempts} attempts")
        print("Will use all available pairs.")
    
    print(f"\nGenerated {len(selected_pairs)} pairs:")
    for i, (list_a, list_b, category_a, category_b) in enumerate(selected_pairs):
        print(f"\nPair {i+1}:")
        print(f"  List A: {list_a} (Category: {category_a})")
        print(f"  List B: {list_b} (Category: {category_b})")
    
    return selected_pairs

if __name__ == "__main__":
    # Initialize analysis
    config_path = Path(__file__).parent.parent / 'config.yaml'
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    analyzer = ActivationPatching('config.yaml')
    
    # Generate list pairs
    list_pairs = generate_list_pairs()
    
    # Create output directory
    output_dir = Path(config['mediation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to: {output_dir}")
    
    # Run analysis for all pairs together
    print(f"\nAnalyzing {len(list_pairs)} pairs")
    results = analyzer.analyze_layer_importance(list_pairs)
    
    # Plot results
    output_path = output_dir / 'layer_importance.png'
    analyzer.plot_layer_importance(results, output_path)
    
    # Save detailed results
    output_file = output_dir / 'mediation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved mediation analysis results to {output_file}")