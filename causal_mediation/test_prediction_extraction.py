import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

def test_prediction_extraction():
    # Load a larger model for testing
    model_name = "EleutherAI/pythia-1.4b"  # Larger model for better reasoning
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test cases with different response formats
    test_cases = [
        {
            "prompt": "Task: Count how many words in a list belong to a specific category.\n\nList: ['cat', 'dog', 'bird', 'fish']\nCategory: animals\n\nCount the number of words that belong to the category 'animals'.\nYour answer should be a single number.\n\nAnswer:",
            "expected": 4,
            "description": "Simple case with clear number"
        },
        {
            "prompt": "Task: Count how many words in a list belong to a specific category.\n\nList: ['apple', 'banana', 'car', 'house']\nCategory: fruits\n\nCount the number of words that belong to the category 'fruits'.\nYour answer should be a single number.\n\nAnswer:",
            "expected": 2,
            "description": "Case with some matching words"
        },
        {
            "prompt": "Task: Count how many words in a list belong to a specific category.\n\nList: ['computer', 'phone', 'tablet', 'laptop']\nCategory: electronics\n\nCount the number of words that belong to the category 'electronics'.\nYour answer should be a single number.\n\nAnswer:",
            "expected": 4,
            "description": "Case with all matching words"
        }
    ]
    
    # Try different generation parameters
    generation_configs = [
        {
            "name": "Default config",
            "params": {
                'max_new_tokens': 5,
                'num_return_sequences': 1,
                'pad_token_id': tokenizer.eos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'forced_eos_token_id': tokenizer.eos_token_id,
                'no_repeat_ngram_size': 3,
                'do_sample': True,
                'temperature': 0.1
            }
        },
        {
            "name": "Higher temperature",
            "params": {
                'max_new_tokens': 5,
                'num_return_sequences': 1,
                'pad_token_id': tokenizer.eos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'forced_eos_token_id': tokenizer.eos_token_id,
                'do_sample': True,
                'temperature': 0.7
            }
        },
        {
            "name": "Deterministic",
            "params": {
                'max_new_tokens': 5,
                'num_return_sequences': 1,
                'pad_token_id': tokenizer.eos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'forced_eos_token_id': tokenizer.eos_token_id,
                'do_sample': False
            }
        }
    ]
    
    print("\nTesting prediction extraction:")
    print("-" * 50)
    
    for config in generation_configs:
        print(f"\nTesting with {config['name']}:")
        print("-" * 30)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest case {i}: {test_case['description']}")
            print(f"Prompt: {test_case['prompt']}")
            
            # Tokenize input
            inputs = tokenizer(test_case['prompt'], return_tensors="pt").to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(**inputs, **config['params'])
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"Raw response: {response}")
            
            # Extract prediction
            try:
                # Find where the prompt ends in the response
                prompt_words = test_case['prompt'].split()
                response_words = response.split()
                
                # Find the prompt end index
                prompt_end_idx = -1
                for i in range(len(response_words) - len(prompt_words) + 1):
                    if response_words[i:i+len(prompt_words)] == prompt_words:
                        prompt_end_idx = i + len(prompt_words)
                        break
                
                if prompt_end_idx == -1:
                    prediction = -1
                    print("Error: Couldn't find where the prompt ends")
                else:
                    # Look for the first number in the response (after the prompt)
                    for word in response_words[prompt_end_idx:]:
                        # Remove any punctuation and try to convert to int
                        clean_word = ''.join(c for c in word if c.isdigit())
                        if clean_word:
                            prediction = int(clean_word)
                            break
                    else:
                        prediction = -1
                        print("Error: No number found in the response")
                
                print(f"Extracted prediction: {prediction}")
                print(f"Expected prediction: {test_case['expected']}")
                print(f"Match: {prediction == test_case['expected']}")
                
            except Exception as e:
                print(f"Error extracting prediction: {str(e)}")
                prediction = -1
            
            print("-" * 30)
    
    # Save test results
    results = {
        "model": model_name,
        "test_cases": test_cases,
        "generation_configs": generation_configs
    }
    
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "prediction_extraction_test.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to {output_file}")

if __name__ == "__main__":
    test_prediction_extraction() 