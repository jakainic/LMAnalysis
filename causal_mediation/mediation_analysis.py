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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Force FP32 for compatibility
            device_map="auto",
            output_hidden_states=True  # We need hidden states for analysis
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Store hooks for activation patching
        self.hooks = []
        self.activations = {}
        
    def _get_activation_hook(self, layer_idx: int):
        """Create a hook to store activations for a specific layer."""
        def hook(module, input, output):
            self.activations[f"layer_{layer_idx}"] = output.detach()
        return hook
    
    def register_hooks(self, layer_idx: int):
        """Register hooks for a specific layer."""
        # Clear previous hooks
        self.remove_hooks()
        
        # Get the layer we want to patch (BLOOM specific)
        layer = self.model.transformer.h[layer_idx]
        
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
        return f"""Given this list of words: {words}
How many words in this list belong to the category '{category}'?
Answer with just a number."""
    
    def get_model_prediction(self, prompt: str) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Get prediction and activations from the model."""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get prediction
        logits = outputs.logits[0]
        # Get the most likely token IDs
        token_ids = logits.argmax(dim=-1)
        # Ensure token_ids is a tensor
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids)
        response = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        try:
            # Find the last number in the response
            words = response.split()
            for word in reversed(words):
                if word.isdigit():
                    prediction = int(word)
                    break
            else:
                prediction = -1
        except:
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
                             category: str,
                             layer_idx: int,
                             token_positions: Optional[List[int]] = None) -> Dict:
        """Run mediation analysis for a specific layer and token positions."""
        # Get original predictions and activations
        prompt_a = self.format_prompt(list_a, category)
        prompt_b = self.format_prompt(list_b, category)
        
        pred_a, acts_a = self.get_model_prediction(prompt_a)
        pred_b, acts_b = self.get_model_prediction(prompt_b)
        
        # Calculate total effect
        total_effect = abs(pred_b - pred_a)
        
        # Patch activations and get patched prediction
        patched_acts = self.patch_activations(acts_a, acts_b, layer_idx, token_positions)
        
        # Run model with patched activations
        inputs = self.tokenizer(prompt_b, return_tensors="pt").to(self.model.device)
        
        # Create a custom forward pass that uses patched activations
        def custom_forward(*args, **kwargs):
            outputs = self.model.transformer(*args, **kwargs)
            # Convert hidden states tuple to list for modification
            hidden_states = list(outputs.hidden_states)
            # Replace the layer's activations
            hidden_states[layer_idx] = patched_acts[f"layer_{layer_idx}"]
            # Convert back to tuple
            outputs.hidden_states = tuple(hidden_states)
            return outputs
        
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
        
        # Calculate direct and indirect effects
        direct_effect = abs(patched_pred - pred_a)
        indirect_effect = abs(pred_b - patched_pred)
        
        # Calculate proportion mediated
        proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0
        
        return {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'proportion_mediated': proportion_mediated,
            'original_predictions': {
                'list_a': pred_a,
                'list_b': pred_b,
                'patched': patched_pred
            }
        }
    
    def analyze_layer_importance(self, 
                               list_pairs: List[Tuple[List[str], List[str]]],
                               category: str,
                               token_positions: Optional[List[int]] = None) -> Dict:
        """Analyze the importance of each layer for running count representation."""
        results = {}
        
        # Get number of layers (BLOOM specific)
        num_layers = len(self.model.transformer.h)
        
        for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
            layer_results = []
            
            for list_a, list_b in list_pairs:
                result = self.run_mediation_analysis(
                    list_a, list_b, category, layer_idx, token_positions
                )
                layer_results.append(result)
            
            # Aggregate results for this layer
            results[layer_idx] = {
                'mean_proportion_mediated': np.mean([r['proportion_mediated'] for r in layer_results]),
                'std_proportion_mediated': np.std([r['proportion_mediated'] for r in layer_results]),
                'mean_indirect_effect': np.mean([r['indirect_effect'] for r in layer_results]),
                'std_indirect_effect': np.std([r['indirect_effect'] for r in layer_results]),
                'individual_results': layer_results
            }
        
        return results
    
    def plot_layer_importance(self, results: Dict, output_path: str):
        """Plot the importance of each layer for running count representation."""
        layers = list(results.keys())
        proportions = [results[layer]['mean_proportion_mediated'] for layer in layers]
        stds = [results[layer]['std_proportion_mediated'] for layer in layers]
        
        plt.figure(figsize=(12, 6))
        plt.bar(layers, proportions, yerr=stds, capsize=5)
        plt.xlabel('Layer Index')
        plt.ylabel('Mean Proportion Mediated')
        plt.title('Layer Importance for Running Count Representation')
        plt.savefig(output_path)
        plt.close()

def generate_similar_list_pairs(num_pairs: int = 10, same_category: bool = True) -> List[Tuple[List[str], List[str], str]]:
    """Generate pairs of similar word lists from existing examples.
    
    Args:
        num_pairs: Number of pairs to generate
        same_category: If True, both lists in a pair will be from the same category
    
    Returns:
        List of tuples containing (list_a, list_b, category)
    """
    # Load examples from the data file
    data_path = Path('data/examples.json')
    with open(data_path, 'r') as f:
        examples = json.load(f)
    
    # Load config to get allowed categories
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    allowed_categories = config['mediation']['categories']
    
    # Group examples by category, filtering for allowed categories
    examples_by_category = defaultdict(list)
    for ex in examples:
        if ex['category'] in allowed_categories:
            examples_by_category[ex['category']].append(ex)
    
    # Randomly sample categories
    allowed_categories = [cat for cat in allowed_categories 
                          if (cat in examples_by_category) and (len(examples_by_category[cat]) >= 2)]
    selected_categories = random.sample(allowed_categories, num_pairs)
    pairs = []
    
    print(f"\nGenerating pairs from categories: {selected_categories}")
    
    # Generate one pair for each selected category
    for category in selected_categories:
        category_examples = examples_by_category[category]
            
        # Find a pair with similar list lengths but different target counts
        for i in range(len(category_examples)):
            ex1 = category_examples[i]
            for j in range(i + 1, len(category_examples)):
                ex2 = category_examples[j]
                if abs(len(ex1['words']) - len(ex2['words'])) <= 2 and ex1['num_target'] != ex2['num_target']:
                    pairs.append((ex1['words'], ex2['words'], category))
                    break
            if len(pairs) > i:  # If we found a pair, break the outer loop
                break
    
    print(f"\nGenerated {len(pairs)} similar list pairs")
    for i, (list_a, list_b, category) in enumerate(pairs):
        print(f"\nPair {i+1} (Category: {category}):")
        print(f"List A: {list_a}")
        print(f"List B: {list_b}")
    
    return pairs

if __name__ == "__main__":
    # Initialize analysis
    analyzer = ActivationPatching('config.yaml')
    
    # Load config to get number of pairs
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    num_pairs = config['mediation']['num_list_pairs']
    
    # Generate similar list pairs (using same category)
    list_pairs = generate_similar_list_pairs(num_pairs=num_pairs, same_category=True)
    
    # Run analysis for each pair using its specific category
    all_results = {}
    for list_a, list_b, category in list_pairs:
        print(f"\nAnalyzing pair with category: {category}")
        results = analyzer.analyze_layer_importance([(list_a, list_b)], category=category)
        all_results[category] = results
    
    # Plot results for each category
    for category, results in all_results.items():
        output_path = Path(f'causal_mediation/layer_importance_{category}.png')
        analyzer.plot_layer_importance(results, output_path)
    
    # Save detailed results
    output_file = Path('causal_mediation/mediation_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2) 