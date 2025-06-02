import requests
from typing import Dict, List, Set
import os
from pathlib import Path

class WordListLoader:
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/imsky/wordlists/master"
        self.api_url = "https://api.github.com/repos/imsky/wordlists/contents"
        self.word_list_types = ['nouns', 'verbs', 'adjectives']
        self.cache_dir = Path('data_generation/cache')
        self.cache_dir.mkdir(exist_ok=True)
    
    def discover_categories(self) -> Dict[str, Set[str]]:
        """Discover all available categories from the word lists repository."""
        discovered_categories = {word_list_type: set() for word_list_type in self.word_list_types}
        
        for word_list_type in self.word_list_types:
            try:
                # Get the directory listing using GitHub API
                url = f"{self.api_url}/{word_list_type}"
                print(f"Fetching directory listing from: {url}")
                response = requests.get(url)
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    # Parse the JSON response to find .txt files
                    files = response.json()
                    for file in files:
                        if file['type'] == 'file' and file['name'].endswith('.txt'):
                            # Extract category name from filename
                            category = file['name'].replace('.txt', '')
                            discovered_categories[word_list_type].add(category)
                            print(f"Found category: {category} in {word_list_type}")
                else:
                    print(f"Failed to fetch directory listing. Status code: {response.status_code}")
                    print(f"Response content: {response.text[:200]}")  # Print first 200 chars of response
            except Exception as e:
                print(f"Warning: Could not fetch directory listing for {word_list_type}: {str(e)}")
        
        return discovered_categories
    
    def load_words_for_category(self, category: str, word_list_type: str) -> List[str]:
        """Load words for a category from a specific word list type."""
        try:
            # Try to load from cache first
            cache_file = self.cache_dir / f"{category}_{word_list_type}.txt"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read().strip().split('\n')
            
            # If not in cache, fetch from repository
            url = f"{self.base_url}/{word_list_type}/{category}.txt"
            print(f"Fetching words from: {url}")
            response = requests.get(url)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                words = response.text.strip().split('\n')
                print(f"Loaded {len(words)} words for {category} ({word_list_type})")
                
                # Cache the words
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(words))
                return words
            elif response.status_code == 404:
                print(f"Category {category} not found in {word_list_type}")
                return []
            else:
                print(f"Failed to fetch words. Status code: {response.status_code}")
                print(f"Response content: {response.text[:200]}")  # Print first 200 chars of response
                return []
        except Exception as e:
            print(f"Warning: Could not fetch words for {category} ({word_list_type}): {str(e)}")
            return []
    
    def load_all_categories(self) -> Dict[str, List[str]]:
        """Load words for all discovered categories."""
        category_words = {}
        discovered_categories = self.discover_categories()
        
        # Get all unique categories across all word list types
        all_categories = set()
        for categories in discovered_categories.values():
            all_categories.update(categories)
        
        for category in all_categories:
            words = []
            # Try to load words from each word list type
            for word_list_type in self.word_list_types:
                if category in discovered_categories[word_list_type]:
                    new_words = self.load_words_for_category(category, word_list_type)
                    words.extend(new_words)
            
            if words:
                # Clean and deduplicate words
                words = [word.strip().lower() for word in words if word.strip()]
                words = list(set(words))  # Remove duplicates
                category_words[category] = words
                print(f"Total words for {category}: {len(words)}")
            else:
                print(f"Warning: No words found for category: {category}")
        
        if not category_words:
            raise ValueError("No word lists could be loaded. Please check your internet connection and the repository availability.")
        
        print(f"\nSuccessfully loaded {len(category_words)} categories:")
        for category, words in category_words.items():
            print(f"- {category}: {len(words)} words")
        
        return category_words

if __name__ == "__main__":
    # Test the loader
    loader = WordListLoader()
    categories = loader.load_all_categories() 