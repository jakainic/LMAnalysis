# Language Model Analysis Pipeline

This project implements a comprehensive pipeline for analyzing language models' behavior in category counting tasks through three main components:

1. Synthetic Data Generation
2. Model Benchmarking
3. Causal Mediation Analysis

## Overview

The pipeline helps understand how language models process and count words belonging to specific categories, with a focus on:
- Generating diverse, controlled test data
- Evaluating model performance across different categories and list lengths
- Analyzing the internal mechanisms through causal mediation

## Components

### 1. Synthetic Data Generation

The data generation component creates controlled test examples for category counting tasks.

**Key Features:**
- Generates word lists with known numbers of target category words
- Implements both systematic (sampling from large, pre-defined category-mediated word lists) and LLM-based generation methods
- Ensures diversity through word frequency tracking and deduplication
- Supports an iterative feedback loop between data generation and validation,
tracking various metrics to ensure a flat distribution over the hyperparameters,
diverse word lists and category choices, and accurate counting of target words.

**Considerations:**
- The validation pipeline uses a hybrid systematic and LLM-based approach, where the systematic approach is tried first and the LLM approach is used if the systematic approach is insufficient to verify the  count. Note that if you are using LLM-generated data, you are advised to use a different LLM for validation to avoid bias.
- Data generation and validation are designed to be iterative, so the validation results may inform hyperparameter tweaks and pruning choices on the previously generated data. I initially wrote an automated script for this iterative loop, but it didn't feel like a marked improvement over a more manual approach.

**Limitations and Room for Improvement:**
- I have chosen to rely on the systematic prompt generation over the LLM prompt generation. Initially, I had sought to include both, with the majority of the data coming from systematic generation (for higher accuracy, better control over distributions, faster generation, a more narrowly defined task) with some portion coming from LLM generation as an "exploration set" (for more variety in category types and word selection, greater exploration of edge cases and challenges like ambiguity, etc.) However, iterative testing of the LLM generation revealed some difficulty assuring accuracy of target word count, and a tendency to choose the same categories again and again. I tried various methods for fixing these problems (adjusting temperature, using pruning strategies to remove bad data, prompt engineering, etc.), but none proved sufficient to make the LLM-generated data high enough quality to use. Given limited time, I set that aside for the time being.
- The systematically generated data generatees data of low word diversity, which is not ideal. I implemented methods to remove duplicates and encourage diversity, but further pushes in this direction would be preferrable. It may be prudent to source word lists from other sources. I had initially tried using WordNet, but generating lists from hyponyms yielded unusual, hard-to-parse results. The pre-determined word lists were more efficient, easier to verify, and higher quality, but are also more limited in their scope.
- It may be advisable to track count-stratified accuracy or list-length-stratified accuracy. 


### 2. Model Benchmarking

The benchmarking component evaluates model performance on the generated test data.

**Key Features:**
- Tests models across different categories and list lengths
- Tracks response distributions and error patterns
- Analyzes performance on systematic vs. LLM-generated examples
- Provides detailed metrics

**Considerations:**
- Typically, synthetic data is not used for testing, so we are taking a somewhat unusual approach here. If using LLM-generated and LLM-validated data, it is highly advised to not benchmark those LLMs on that data, to avoid biased results. While my code has capability to use both LLM and systematically generated data, for benchmarking I have chosen only to use the systematic data, as it yields less ambiguous prompts (hence easier to benchmark accuracy).

**Results, Limitations, and Room for Improvement:**
- BIG ISSUES: Clearly, looking at my results, something is very wrong here. The Pythia and Opt models have identical performance, with extremely low accuracy (3.4%) and only correct predictions on cases where the list length = number of target words = 5. On the other hand, the Bloom model has a slightly higher but still shockingly low accuracy (17.8%) with a response highly skewed towards 1. All of this information suggests to me that there is a deep problem with my methodology. Unfortunately, I do not have time to sort it out. 
- I have relied mostly on smaller LMs for benchmarking, for effiency and ease of iteration given time limitations. However, testing larger LMs as well would likely give more interesting and varied results.
- Given better LLM-generated data, it would be preferable to not merely benchmark on systematically generated data but also on LLM-generated data, which can provide a great challenge and exploration of edge cases.


### 3. Causal Mediation Analysis

The mediation analysis component investigates how models process category counting through activation patching.

**Key Features:**
- Implements activation patching to analyze model internals
- Tracks direct and indirect effects of interventions
- Analyzes layer importance for each layer for category counting
- Generates visualizations of mediation effects

**Results, Limitations and Room for Improvement**
-BIG ISSUES: Haven't had a chance to run this all the way through.
- My code has the infrastructure for position-sensitive patching, which would be especially useful for answering our question of whether we have a latent representation of a *running* count, but we're not using it in the main analysis. This would be room for great improvement in future iteration.
- Due to time restrictions, I've run the mediation analysis experiment on a small model that is likely less accurate at the task than larger models. I've also run it on very few samples. Therefore, the results are likely markedly less interesting and less revealing in answering our underlying questions than they could be given a better model and more data.
- For the mediation analysis, I chose to compare results from pairs of words that were in some senses similar (same category, similar list lengths) so as to hopefully not introduce confounding factors. However, I didn't think at length about what the data should look like and how to account for potential confounders. I'm sure there are a variety of approaches that could be used to address this problem -- perturbing the same datum (but then the signal may be too weak), systematic variation to control for specific factors (just takes more careful design up front), balanced data sets, etc. The key, I image, is that you want changes in count that aren't well correlated with other features.
- I chose to restrict to a very finite set of categories just for ease of making similar data pairs. Obviously, for a more comprehensive analysis it would be better to draw from a wider variety of categories.

## Usage

**Considerations:**
- Make sure all relevant API keys are accessible.
- Log into huggingface-cli for benchmarking and mediation analysis.

1. **Generate Test Data:**
```bash
python data_generation/data_generation.py
```
Note: this should be iterative, alternating with
```bash
python data_generation/validation_pipeline.py
```, the results of which should inform hyperparameter tweaks and data pruning on the previously generated data.

2. **Run Model Benchmarks:**
```bash
python benchmarking/benchmark_models.py
```

3. **Perform Mediation Analysis:**
```bash
python causal_mediation/mediation_analysis.py
```

## Output

The pipeline generates several types of outputs:

1. **Data Generation:**
   - `data/examples.json`: Generated test examples
   -`validation_results.json`: Summary of validation metrics, and examples of inaccurate data (i.e. count is wrong).

2. **Benchmarking:**
   - `data/benchmark_results.json`: Model performance metrics
   - Performance visualizations by category and list length

3. **Mediation Analysis:**
   - `mediation_analysis/mediation_results.json`: Detailed mediation results
   - `mediation_analysis/layer_importance_*.png`: Layer importance visualizations

## Design Choices

1. **Data Generation:**
   - Mixed systematic and LLM-based generation for diverse examples
   - Word frequency tracking to ensure balanced word usage
   - Jaccard similarity for deduplication

2. **Benchmarking:**
   - Strict prompt formatting to ensure consistent responses
   - Comprehensive metrics tracking
   - Separate analysis of systematic vs. LLM examples

3. **Mediation Analysis:**
   - Layer-wise activation patching
   - Proportion mediated as key metric
   - Visualization of layer importance

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NLTK
- NumPy
- Matplotlib
- Seaborn

## Installation

```bash
pip install -r requirements.txt
```