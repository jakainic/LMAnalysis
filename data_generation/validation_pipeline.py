import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests
from collections import Counter
import yaml
import anthropic
import openai
import os
import numpy as np
from word_list_loader import WordListLoader

class ValidationPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize validation LLM client
        self.llm_client = None
        if self.config['validation']['model'].startswith('claude'):
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
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def validate_with_wordlists(self, words: List[str], category: str) -> Tuple[int, bool]:
        """Validate words using word lists."""
        if category not in self.category_words:
            return 0, False  # Can't validate with word lists
        
        target_count = sum(1 for word in words if word in self.category_words[category])
        return target_count, True
    
    def validate_with_llm(self, words: List[str], category: str, num_target: int) -> int:
        """Validate words using LLM with balanced categorization."""
        prompt = f"""Given the following list of words: {words}
        How many words in this list belong to the category '{category}'?
        
        Guidelines:
        - Count words that belong to the category in ANY of their meanings
        - A word can belong to the category even if it has other meanings
        - At least one of the meanings much clearly belong to the category, but not all meanings must belong to the category to count
        
        Return only the numerical count."""
        
        if self.config['validation']['model'].startswith('claude'):
            response = self.llm_client.messages.create(
                model=self.config['validation']['model'],
                max_tokens=self.config['validation']['max_tokens'],
                temperature=self.config['validation']['temperature'],
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text.strip()
        else:  # OpenAI
            response = self.llm_client.chat.completions.create(
                model=self.config['validation']['model'],
                max_completion_tokens=self.config['validation']['max_tokens'],
                temperature=self.config['validation']['temperature'],
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip()
        
        try:
            count = int(response_text)
            return count
        except ValueError:
            print(f"Warning: LLM returned non-numeric response: '{response_text}'")
            # If we can't parse the response as a number, return 0 to mark as invalid
            return 0
    
    def validate_example(self, example: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a single example using hybrid approach."""
        category = example['category']
        words = example['words']
        num_target = example['num_target']
        method = example['generation_method']
        
        # Try systematic validation first
        target_count, can_validate = self.validate_with_wordlists(words, category)
        
        if can_validate:
            # If we can validate systematically and the count matches, we're done
            if target_count == num_target:
                return True, "Validated using word lists"
            
        # If count doesn't match or can't validate, use LLM as second opinion
        llm_count = self.validate_with_llm(words, category, num_target)
        
        if llm_count == num_target:
            return True, "Validated using LLM"
        else:
            return False, f"LLM validation found {llm_count} target words, expected {num_target}"
    
    def _calculate_metrics_for_subset(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for a subset of examples."""
        if not examples:
            return {}
        
        # Calculate accuracy and collect explanations
        valid_examples = []
        invalid_examples = []
        for ex in examples:
            is_valid, explanation = self.validate_example(ex)
            if is_valid:
                valid_examples.append(ex)
            else:
                invalid_examples.append(ex)
        
        accuracy = len(valid_examples) / len(examples)
        
        # Calculate other metrics
        category_counts = Counter(ex['category'] for ex in examples)
        category_distribution = {cat: count/len(examples) for cat, count in category_counts.items()}
        
        length_counts = Counter(len(ex['words']) for ex in examples)
        length_distribution = {length: count/len(examples) for length, count in length_counts.items()}
        
        target_counts = Counter(ex['num_target'] for ex in examples)
        target_distribution = {count: freq/len(examples) for count, freq in target_counts.items()}
        
        # Calculate word diversity and duplicate rate at list level
        all_words = [word for ex in examples for word in ex['words']]
        unique_words = set(all_words)
        word_diversity = len(unique_words) / len(all_words) if all_words else 0
        
        # Calculate duplicate rate at list level
        word_lists = [tuple(sorted(ex['words'])) for ex in examples]  # Sort to ensure consistent ordering
        list_counts = Counter(word_lists)
        duplicate_rate = sum(count - 1 for count in list_counts.values()) / len(examples) if examples else 0
        
        # Debug logging
        print(f"\nDebug - Word Diversity Analysis:")
        print(f"Total words: {len(all_words)}")
        print(f"Unique words: {len(unique_words)}")
        print(f"Word diversity score: {word_diversity:.3f}")
        print(f"Number of unique word lists: {len(list_counts)}")
        print(f"Most common word lists and their counts: {list_counts.most_common(3)}")
        print(f"Duplicate rate (list level): {duplicate_rate:.3f}")
        
        return {
            'accuracy': accuracy,
            'category_distribution': category_distribution,
            'list_length_distribution': length_distribution,
            'target_count_distribution': target_distribution,
            'word_diversity': word_diversity,
            'duplicate_rate': duplicate_rate,
            'validation_details': {
                'valid_examples': valid_examples,
                'invalid_examples': invalid_examples
            }
        }
    
    def calculate_metrics(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate validation metrics for the dataset."""
        # Split examples by generation method
        systematic_examples = [ex for ex in examples if ex['generation_method'] == 'systematic']
        llm_examples = [ex for ex in examples if ex['generation_method'] == 'llm']
        
        # Calculate metrics for each subset
        systematic_metrics = self._calculate_metrics_for_subset(systematic_examples)
        llm_metrics = self._calculate_metrics_for_subset(llm_examples)
        combined_metrics = self._calculate_metrics_for_subset(examples)
        
        return {
            'systematic': {
                'count': len(systematic_examples),
                'metrics': systematic_metrics
            },
            'llm': {
                'count': len(llm_examples),
                'metrics': llm_metrics
            },
            'combined': {
                'count': len(examples),
                'metrics': combined_metrics
            }
        }
    
    def analyze_distributions(self, examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze distributions of categories, list lengths, and target counts."""
        # Separate examples by generation method
        systematic_examples = [ex for ex in examples if ex['generation_method'] == 'systematic']
        llm_examples = [ex for ex in examples if ex['generation_method'] == 'llm']
        
        results = {
            'systematic': self._analyze_subset(systematic_examples, 'systematic'),
            'llm': self._analyze_subset(llm_examples, 'llm')
        }
        
        return results
    
    def _analyze_subset(self, examples: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
        """Analyze distributions for a subset of examples."""
        # Category distribution
        categories = [ex['category'] for ex in examples]
        category_counts = Counter(categories)
        expected_category_freq = 1.0 / len(self.categories)  # Use discovered categories
        category_uniformity = self._calculate_uniformity(category_counts.values(), expected_category_freq)
        
        # Track invalid examples by category
        invalid_by_category = Counter()
        for ex in examples:
            is_valid, _ = self.validate_example(ex)
            if not is_valid:
                invalid_by_category[ex['category']] += 1
        
        # Calculate invalid percentage by category
        invalid_percentages = {
            cat: (invalid_by_category[cat] / category_counts[cat]) * 100 
            for cat in category_counts.keys()
        }
        
        # List length distribution
        lengths = [len(ex['words']) for ex in examples]
        length_counts = Counter(lengths)
        expected_length_freq = 1.0 / (self.config['max_list_length'] - self.config['min_list_length'] + 1)
        length_uniformity = self._calculate_uniformity(length_counts.values(), expected_length_freq)
        
        # Target count distribution
        targets = [ex['num_target'] for ex in examples]
        target_counts = Counter(targets)
        max_possible_targets = min(self.config['max_target_words'], self.config['max_list_length'])
        min_possible_targets = self.config['min_target_words']
        expected_target_freq = 1.0 / (max_possible_targets - min_possible_targets + 1)
        target_uniformity = self._calculate_uniformity(target_counts.values(), expected_target_freq)
        
        return {
            'category_distribution': {
                'counts': dict(category_counts),
                'uniformity_score': category_uniformity,
                'invalid_percentages': invalid_percentages
            },
            'length_distribution': {
                'counts': dict(length_counts),
                'uniformity_score': length_uniformity
            },
            'target_distribution': {
                'counts': dict(target_counts),
                'uniformity_score': target_uniformity
            }
        }
    
    def _calculate_uniformity(self, counts: List[int], expected_freq: float) -> float:
        """Calculate uniformity score using chi-square test.
        
        Args:
            counts: List of counts for each category
            expected_freq: Expected frequency for each category
            
        Returns:
            float: Uniformity score between 0 and 1, where 1 is perfectly uniform
        """
        if not counts:  # Handle case where there are no examples
            return 0.0
            
        total = sum(counts)
        if total == 0:  # Handle case where all counts are 0
            return 0.0
            
        # Calculate expected counts based on total
        expected_counts = [total * expected_freq] * len(counts)
        
        # Calculate chi-square statistic
        chi_square = sum((obs - exp) ** 2 / exp for obs, exp in zip(counts, expected_counts))
        
        # Calculate maximum possible chi-square (all observations in one category)
        max_chi_square = total * (1 - expected_freq) ** 2 / expected_freq
        
        if max_chi_square == 0:  # Handle case where all counts are equal
            return 1.0
            
        # Convert to a score between 0 and 1, where 1 is perfectly uniform
        uniformity = 1 - (chi_square / max_chi_square)
        
        # Ensure the score is between 0 and 1
        return max(0.0, min(1.0, uniformity))
    
    def validate_dataset(self, input_file: str = None):
        """Validate the entire dataset and analyze distributions."""
        if input_file is None:
            input_file = self.output_dir / self.config['data_file']
        
        with open(input_file, 'r') as f:
            examples = json.load(f)
        
        # First analyze distributions
        print("\nAnalyzing distributions...")
        distribution_results = self.analyze_distributions(examples)
        
        # Print distribution analysis results
        print("\nDistribution Analysis Results:")
        for method in ['systematic', 'llm']:
            print(f"\n{method.upper()} GENERATION:")
            for metric in ['category', 'length', 'target']:
                uniformity = distribution_results[method][f'{metric}_distribution']['uniformity_score']
                print(f"{metric.title()} Distribution Uniformity Score: {uniformity:.3f}")
        
        # Then perform validation
        print("\nValidating examples...")
        invalid_examples = []
        valid_count = 0
        
        # Track validation results by generation method
        systematic_results = {'valid': 0, 'invalid': 0}
        llm_results = {'valid': 0, 'invalid': 0}
        
        for example in examples:
            is_valid, explanation = self.validate_example(example)
            method = example['generation_method']
            
            if is_valid:
                valid_count += 1
                if method == 'systematic':
                    systematic_results['valid'] += 1
                else:
                    llm_results['valid'] += 1
            else:
                invalid_examples.append((example, explanation))
                if method == 'systematic':
                    systematic_results['invalid'] += 1
                else:
                    llm_results['invalid'] += 1
        
        # Calculate accuracies
        systematic_total = systematic_results['valid'] + systematic_results['invalid']
        llm_total = llm_results['valid'] + llm_results['invalid']
        
        systematic_accuracy = systematic_results['valid'] / systematic_total if systematic_total > 0 else 0
        llm_accuracy = llm_results['valid'] / llm_total if llm_total > 0 else 0
        overall_accuracy = valid_count / len(examples)
        
        # Calculate word diversity metrics
        systematic_examples = [ex for ex in examples if ex['generation_method'] == 'systematic']
        llm_examples = [ex for ex in examples if ex['generation_method'] == 'llm']
        
        systematic_metrics = self._calculate_metrics_for_subset(systematic_examples)
        llm_metrics = self._calculate_metrics_for_subset(llm_examples)
        combined_metrics = self._calculate_metrics_for_subset(examples)
        
        # Print validation results
        print(f"\nValidation Results:")
        print(f"Total examples: {len(examples)}")
        print(f"Valid examples: {valid_count}")
        print(f"Invalid examples: {len(invalid_examples)}")
        
        print("\nAccuracy Summary:")
        print(f"Overall accuracy: {overall_accuracy:.1%}")
        print(f"Systematic generation accuracy: {systematic_accuracy:.1%} ({systematic_results['valid']}/{systematic_total})")
        print(f"LLM generation accuracy: {llm_accuracy:.1%} ({llm_results['valid']}/{llm_total})")
        
        print("\nWord Diversity Metrics:")
        print(f"Systematic generation:")
        print(f"  Word diversity: {systematic_metrics.get('word_diversity', 0.0):.3f}")
        print(f"  Duplicate rate: {systematic_metrics.get('duplicate_rate', 0.0):.3f}")
        print(f"LLM generation:")
        print(f"  Word diversity: {llm_metrics.get('word_diversity', 0.0):.3f}")
        print(f"  Duplicate rate: {llm_metrics.get('duplicate_rate', 0.0):.3f}")
        print(f"Combined:")
        print(f"  Word diversity: {combined_metrics.get('word_diversity', 0.0):.3f}")
        print(f"  Duplicate rate: {combined_metrics.get('duplicate_rate', 0.0):.3f}")
        
        # Save validation results
        results = {
            'distribution_analysis': distribution_results,
            'validation_results': {
                'total_examples': len(examples),
                'valid_examples': valid_count,
                'invalid_examples': len(invalid_examples),
                'accuracy_summary': {
                    'overall': overall_accuracy,
                    'systematic': {
                        'accuracy': systematic_accuracy,
                        'valid': systematic_results['valid'],
                        'total': systematic_total
                    },
                    'llm': {
                        'accuracy': llm_accuracy,
                        'valid': llm_results['valid'],
                        'total': llm_total
                    }
                },
                'word_diversity_metrics': {
                    'systematic': {
                        'word_diversity': systematic_metrics.get('word_diversity', 0.0),
                        'duplicate_rate': systematic_metrics.get('duplicate_rate', 0.0)
                    },
                    'llm': {
                        'word_diversity': llm_metrics.get('word_diversity', 0.0),
                        'duplicate_rate': llm_metrics.get('duplicate_rate', 0.0)
                    },
                    'combined': {
                        'word_diversity': combined_metrics.get('word_diversity', 0.0),
                        'duplicate_rate': combined_metrics.get('duplicate_rate', 0.0)
                    }
                },
                'invalid_examples_details': [
                    {
                        'category': ex['category'],
                        'words': ex['words'],
                        'num_target': ex['num_target'],
                        'generation_method': ex['generation_method'],
                        'explanation': exp
                    }
                    for ex, exp in invalid_examples
                ]
            }
        }
        
        with open(self.output_dir / self.config['files']['validation_results'], 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nValidation results saved to {self.output_dir / self.config['files']['validation_results']}")

if __name__ == "__main__":
    validator = ValidationPipeline('config.yaml')
    validator.validate_dataset()

