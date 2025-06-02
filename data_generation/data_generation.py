import yaml
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import requests
import anthropic
import openai
import os
from tqdm import tqdm
from collections import Counter
from word_list_loader import WordListLoader

class DataGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize only the specified LLM client
        self.llm_client = None
        if self.config['llm']['model'].startswith('claude'):
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.llm_client = anthropic.Anthropic(api_key=api_key)
        else:  # OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.llm_client = openai.OpenAI(api_key=api_key)
        
        # Initialize word list loader and load categories
        self.word_list_loader = WordListLoader()
        self.category_words = self.word_list_loader.load_all_categories()
        self.categories = list(self.category_words.keys())
        
        # Initialize word frequency tracking
        self.word_frequencies = {}
        for category, words in self.category_words.items():
            for word in words:
                self.word_frequencies[word] = 0
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize category tracking
        self.category_counts = {}
        self.used_custom_categories = set()
        self.max_category_usage_percent = 0.10  # Maximum percentage of total examples a category can be used

    def generate_systematic_example(self) -> Tuple[str, List[str], int]:
        """Generate an example using pre-defined word lists."""
        # Randomly select category
        category = random.choice(self.categories)
        
        # Determine list length and number of target words
        list_length = random.randint(
            self.config['min_list_length'],
            self.config['max_list_length']
        )
        num_target = random.randint(
            self.config['min_target_words'],
            min(self.config['max_target_words'], list_length)
        )
        
        # Get words for the selected category, with a gentle preference for less frequently used words
        available_words = self.category_words[category]
        # Use a gentler scoring formula: 1.0 / (1 + frequency) instead of 1.0 / (frequency + 1)
        word_scores = [(word, 1.0 / (1 + self.word_frequencies.get(word, 0))) for word in available_words]
        # Sort by score but add some randomness to avoid always picking the same words
        word_scores.sort(key=lambda x: x[1] * random.uniform(0.8, 1.2), reverse=True)
        target_words = [word for word, _ in word_scores[:num_target]]
        
        # Update word frequencies for target words
        for word in target_words:
            self.word_frequencies[word] = self.word_frequencies.get(word, 0) + 1
        
        # Get random words from other categories for the remaining slots
        other_words = []
        other_categories = [c for c in self.categories if c != category]
        for _ in range(list_length - num_target):
            other_category = random.choice(other_categories)
            available_words = self.category_words[other_category]
            # Use the same gentle scoring for other words
            word_scores = [(word, 1.0 / (1 + self.word_frequencies.get(word, 0))) for word in available_words]
            word_scores.sort(key=lambda x: x[1] * random.uniform(0.8, 1.2), reverse=True)
            other_word = word_scores[0][0]
            other_words.append(other_word)
            self.word_frequencies[other_word] = self.word_frequencies.get(other_word, 0) + 1
        
        # Combine and shuffle words
        all_words = target_words + other_words
        random.shuffle(all_words)
        
        return category, all_words, num_target
    
    def _get_available_categories(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Get list of categories that are available for use based on usage limits.
        
        Args:
            examples: List of existing examples
            target_num_examples: Target number of examples to generate. If None, uses config value.
        """
        target_num_examples = self.config['num_examples']
            
        max_allowed_usage = max(1, int(target_num_examples * self.max_category_usage_percent))
        available = []
        
        # Get all categories that haven't reached their usage limit
        for category in self.categories:
            current_usage = self.category_counts.get(category, 0)
            if current_usage >= max_allowed_usage:
                print(f"Category '{category}' has reached max usage ({current_usage}/{max_allowed_usage})")
                continue
            available.append(category)
            
        if not available:
            print("No categories available, resetting counts")
            self.category_counts = {}
            return self.categories

        return available

    def generate_llm_example(self) -> Tuple[str, List[str], int, str]:
        """Generate an example using LLM with focus on exploration and edge cases."""
        max_retries = 5  # Maximum number of retries for invalid JSON
        retry_count = 0
        
        # Determine list length and number of target words from config ranges
        list_length = random.randint(
            self.config['min_list_length'],
            self.config['max_list_length']
        )
        num_target = random.randint(
            self.config['min_target_words'],
            min(self.config['max_target_words'], list_length)
        )
        
        while retry_count < max_retries:
            try:
                # Determine if LLM should choose its own category
                use_custom_category = random.random() < self.config['llm']['custom_category_probability']
                
                if use_custom_category:
                    # Avoid most used categories to encourage diversity
                    most_used_categories = sorted(
                        self.category_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    # Let LLM choose its own category
                    prompt = f"""Generate a list of exactly {list_length} words where exactly {num_target} words belong to a specific category.
                    Choose a well-defined category with clear boundaries and consistent criteria for membership.
                    You may choose a category that is more abstract or complex for an added challenge, but it should still be feasible
                    to generate a list of words with a well-defined subset of words belonging to the category.
                    
                    Important requirements:
                    1. The list MUST contain exactly {list_length} words
                    2. The number of target words MUST exactly match {num_target}
                    3. Choose diverse words that are clearly in the category, though you may include edge cases
                    4. Avoid repetitive or very similar words
                    5. Use consistent criteria for category membership
                    6. You MUST return valid JSON with no additional text
                    7. Choose a unique category with clear, objective membership criteria
                    8. Avoid recently used categories like: {', '.join(self.used_custom_categories)}
                    9. Avoid these overrepresented categories: {', '.join(cat for cat, _ in most_used_categories)}
                    10. Consider categories like:
                       - Professional roles in a specific industry
                       - Musical genres or styles
                       - Literary devices or techniques
                       - Cultural traditions
                       - Historical periods
                       - Artistic movements
                       - Geological formations
                       - Economic concepts
                       - Philosophical ideas
                       - Technological innovations
                    
                    For example, if the category is 'fruit', you might include:
                    - Common fruits (apple, banana)
                    - Less common fruits (durian, persimmon)
                    - Fruits that might be ambiguous (tomato, avocado)
                    - Fruits that might be known by different names (pitaya, dragon fruit)
                    
                    Return the response in JSON format with keys 'category', 'words', 'num_target', and 'notes'.
                    Example format:
                    {{
                        "category": "mythical_creatures",
                        "words": ["dragon", "dog", "phoenix", "cat", "unicorn"],
                        "num_target": 3,
                        "notes": "Included common animals as non-mythical creatures for contrast"
                    }}
                    """
                else:
                    # Use predefined category, preferring less used ones
                    available_categories = self._get_available_categories([])
                    if not available_categories:
                        print("Warning: No categories available for LLM generation")
                        return None, [], 0, "No available categories"
                    
                    # Sort categories by usage count
                    category = min(available_categories, 
                                 key=lambda c: self.category_counts.get(c, 0))
                    
                    prompt = f"""Generate a list of exactly {list_length} words where exactly {num_target} words belong to the category '{category}'.
                    The remaining words should be from other categories.
                    
                    Important requirements:
                    1. The list MUST contain exactly {list_length} words
                    2. The number of target words MUST exactly match {num_target}
                    3. Choose diverse words that are clearly in the category.
                    4. Use consistent criteria for category membership
                    5. Include some interesting or less common examples, but avoid cases that are too hard to categorize definitively
                    6. You MUST return valid JSON with no additional text
                    
                    For example, if the category is 'fruit', you might include:
                    - Common fruits (apple, banana)
                    - Less common fruits (durian, persimmon)
                    - Fruits that might be ambiguous (tomato, avocado)
                    - Fruits that might be known by different names (pitaya, dragon fruit)
                    
                    Return the response in JSON format with keys 'category', 'words', 'num_target', and 'notes'.
                    Example format:
                    {{
                        "category": "fruit",
                        "words": ["apple", "dog", "durian", "cat", "tomato"],
                        "num_target": 3,
                        "notes": "Included tomato which is botanically a fruit but often used as a vegetable"
                    }}"""
                
                if self.config['llm']['model'].startswith('claude'):
                    response = self.llm_client.messages.create(
                        model=self.config['llm']['model'],
                        max_tokens=self.config['llm']['max_tokens'],
                        temperature=self.config['llm']['temperature'],
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result = json.loads(response.content[0].text)
                else:  # OpenAI
                    response = self.llm_client.chat.completions.create(
                        model=self.config['llm']['model'],
                        max_tokens=self.config['llm']['max_tokens'],
                        temperature=self.config['llm']['temperature'],
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result = json.loads(response.choices[0].message.content)
                
                # Validate required fields are present
                required_fields = ['category', 'words', 'num_target']
                if not all(field in result for field in required_fields):
                    raise ValueError("Missing required fields in JSON response")
                
                # Validate list length and target count match our requirements
                if len(result['words']) != list_length:
                    raise ValueError(f"List length {len(result['words'])} does not match required length {list_length}")
                if result['num_target'] != num_target:
                    raise ValueError(f"Target count {result['num_target']} does not match required count {num_target}")
                
                # Update category tracking
                category = result['category']
                self.category_counts[category] = self.category_counts.get(category, 0) + 1
                
                if use_custom_category:
                    self.used_custom_categories.add(category)
                    # Keep only last 10 used categories to avoid prompt getting too long
                    if len(self.used_custom_categories) > 10:
                        self.used_custom_categories.pop()
                
                return category, result['words'], result['num_target'], result.get('notes', '')
                
            except (json.JSONDecodeError, ValueError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Invalid JSON response (attempt {retry_count}/{max_retries}), retrying...")
                    continue
                else:
                    print(f"Failed to generate valid JSON after {max_retries} attempts")
                    # Generate a systematic example as fallback
                    category, words, num_target = self.generate_systematic_example()
                    return category, words, num_target, "Fallback to systematic generation due to LLM JSON errors"
    
    def _is_duplicate(self, example: Dict[str, Any], existing_examples: List[Dict[str, Any]], max_similarity: float = None) -> bool:
        """Check if an example is too similar to existing examples."""
        if max_similarity is None:
            # Use stricter threshold for LLM examples
            if example['generation_method'] == 'llm':
                max_similarity = self.config['deduplication']['max_similarity'] * 0.8
            else:
                max_similarity = self.config['deduplication']['max_similarity']
        
        # Sort words to ensure consistent comparison
        example_words = set(example['words'])
        
        # Check for similar examples using Jaccard similarity
        for existing in existing_examples:
            existing_words = set(existing['words'])
            
            # Calculate Jaccard similarity
            intersection = len(example_words & existing_words)
            union = len(example_words | existing_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > max_similarity:
                return True
        
        return False
    
    def generate_dataset(self, num_examples: int = None):
        """Generate a mixed dataset of systematic and LLM examples."""
        num_examples = self.config['num_examples']
        
        # Get method proportions from config, default to 70/30 if not specified
        method_proportions = self.config.get('method_proportions', {
            'systematic': 0.7,
            'llm': 0.3
        })
        
        # Calculate number of examples for each method
        num_systematic = int(num_examples * method_proportions['systematic'])
        num_llm = num_examples - num_systematic
        
        examples = []
        max_attempts = num_examples * self.config['deduplication']['max_attempts_multiplier']
        attempts = 0
        
        # Generate systematic examples with deduplication
        print(f"\nGenerating {num_systematic} systematic examples...")
        systematic_generated = 0
        while systematic_generated < num_systematic and attempts < max_attempts:
            # Get available categories for systematic generation
            available_categories = self._get_available_categories(examples)
            if not available_categories:
                print("Warning: No categories available for systematic generation")
                break
                
            category, words, num_target = self.generate_systematic_example()
            if category is None:  # Skip invalid examples
                attempts += 1
                continue
                
            example = {
                'category': category,
                'words': words,
                'num_target': num_target,
                'generation_method': 'systematic',
                'notes': ''
            }
            
            if not self._is_duplicate(example, examples):
                examples.append(example)
                systematic_generated += 1
                # Update category tracking
                self.category_counts[category] = self.category_counts.get(category, 0) + 1
                if systematic_generated % 10 == 0:
                    print(f"Generated {systematic_generated}/{num_systematic} systematic examples...")
            attempts += 1
        
        if systematic_generated < num_systematic:
            print(f"Warning: Could only generate {systematic_generated} systematic examples after {attempts} attempts")
        
        # Generate LLM examples with the same deduplication logic
        print(f"\nGenerating {num_llm} LLM examples...")
        llm_generated = 0
        while llm_generated < num_llm and attempts < max_attempts:
            category, words, num_target, notes = self.generate_llm_example()
            if category is None:  # No available categories
                break
                
            example = {
                'category': category,
                'words': words,
                'num_target': num_target,
                'generation_method': 'llm',
                'notes': notes
            }
            
            if not self._is_duplicate(example, examples):
                examples.append(example)
                llm_generated += 1
                if llm_generated % 5 == 0:
                    print(f"Generated {llm_generated}/{num_llm} LLM examples...")
            attempts += 1
        
        if llm_generated < num_llm:
            print(f"Warning: Could only generate {llm_generated} LLM examples after {attempts} attempts")
        
        # Print final category distribution
        print("\nFinal category distribution:")
        for category, count in sorted(self.category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{category}: {count} examples ({count/len(examples)*100:.1f}%)")
        
        # Shuffle all examples
        random.shuffle(examples)
        
        # Save all examples in a single file
        output_file = self.output_dir / 'mixed_examples.json'
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"\nSaved {len(examples)} examples to {output_file}")
        print(f"Breakdown: {systematic_generated} systematic examples, {llm_generated} LLM examples")
        print(f"Total attempts: {attempts}")
        if len(examples) < num_examples:
            print(f"Note: Generated {len(examples)}/{num_examples} examples due to deduplication constraints")

if __name__ == "__main__":
    generator = DataGenerator('config.yaml')
    generator.generate_dataset()



