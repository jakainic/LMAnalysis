# Language Model Analysis Pipeline - QUICK SPRINT

This project implements a comprehensive pipeline for analyzing language models' behavior in category counting tasks through three main components:

1. Synthetic Data Generation
2. Model Benchmarking
3. Causal Mediation Analysis

This project was implemented in 2 days and is therefore not comprehensive nor fully debugged.

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
- I have generated systematic and LLM examples separately, though the code supports generating a mixed data set. Ideally I had sought to produce a mixed dataset, with the majority of the data coming from systematic generation (for higher accuracy, better control over distributions, faster generation, a more narrowly defined task) with some portion coming from LLM generation as an "exploration set" (for more variety in category types and word selection, greater exploration of edge cases and challenges like ambiguity, etc.). Looking at the validation for these two data sets, this proves to be true -- the LLM data has a wider variety of categories and much higher word diversity than the systematic data. However, the accuracy on the LLM data is much lower.
I have done some prompt engineering and hyperparameter fine-tuning to improve this number, but more work is needed to improve either the quality of the LLM data or the validation pipeline.
- The systematically generated data generatees data of low word diversity, which is not ideal. I implemented methods to remove duplicates and encourage diversity, but further pushes in this direction would be preferrable. It may be prudent to source word lists from other sources. I had initially tried using WordNet, but generating lists from hyponyms yielded unusual, hard-to-parse results. The pre-determined word lists were more efficient, easier to verify, and higher quality, but are also more limited in their scope.
- It may be advisable to track count-stratified accuracy or list-length-stratified accuracy. 


### 2. Model Benchmarking

The benchmarking component evaluates model performance on the generated test data.

**Key Features:**
- Tests models across different categories and list lengths
- Tracks response distributions and error patterns
- Analyzes performance on systematic vs. LLM-generated examples
- Provides detailed metrics
- Compatible running locally or in Colab

**Considerations:**
- If using LLM-generated and LLM-validated data, it is highly advised to not benchmark those LLMs on the data in question, to avoid biased results. While my code has capability to use both LLM and systematically generated data, for benchmarking I have chosen only to use the systematic data, as it yields less ambiguous prompts (hence easier to benchmark accuracy).

**Results, Limitations, and Room for Improvement:**
- Biggest issue: Is somewhat unknown. I have relied mostly on smaller LMs for benchmarking, for effiency and ease of iteration given time limitations. After various rounds of benchmarking and trouble-shooting (including parameter tuning, prompt engineering, confirming via a smaller unit test that my prediction extraction is working as intended, etc.) with these smaller models, the best accuracy I got was 0.158, which is still pitifully low. My understanding is that this may not entirely unexpected: smaller models struggle with following directions and may lack the semantic knowledge needed for categorization. That being said, I also ran a benchmark on Mistral Instruct 7b (quantized), and its accuracy was only 0.133. Given that this much larger and more powerful model also struggled significantly, I can only conclude that there is something wrong with my code for running these benchmarks and possibly for extracting predictions more specifically. Frankly, I simply ran out of time to fully target the source of the issue. One interesting observation is that the Mistral model responses were heavily skewed to 0, which may indicate a problem with temperature-setting, prompt engineering, etc.
- Given higher quality LLM-generated data and larger benchmarking models, it would be interesting to not merely benchmark on systematically generated data but also on LLM-generated data, which can provide a great challenge and exploration of edge cases, thereby revealing more about true capabilities.


### 3. Causal Mediation Analysis

The mediation analysis component investigates how models process category counting through activation patching.

**Key Features:**
- Implements activation patching to analyze model internals
- Tracks direct and indirect effects of interventions
- Analyzes layer importance for each layer for category counting
- Generates visualizations of mediation effects
- Compatible running locally or in Colab

**Considerations:**
- As with benchmarking, if using LLM-generated and LLM-validated data, it is highly advised to not use one of those models for causal mediation analysis (not an issue here since the causal mediation analysis can only be done on open weight models, and I've used closed-weight models for data generation and validation). 
- As with benchmarking, while my code has capability to use both LLM and systematically generated data, for benchmarking I have chosen only to use the systematic data, as it yields less ambiguous prompts that might introduce errors that have nothing to do with the causal effect of activation layers.

**Results, Limitations and Room for Improvement**
- I initially ran the mediation analysis on one of the small models, which yielded no causal effects, which is not necessarily surprising given how poorly the small models performed. It would be hard to detect a latent representation in a model that seems to always be choosing the same response and therefore is not sensitive to inputs. 
- I then ran the mediation analysis on the Mistral Instruct 7b v1. Though this model also didn't perform well on the benchmarks, it did detect some very small mediation effects. The mean proportion mediated across layers ranges from -0.025 to +0.05. Early layers show the strongest mediation effects. Middle layers show some higher indirect effects. The variability of both proportion mediated and indirect effect are high, suggesting that the mediation patterns are highly dependent on the specific input cases, and therefore likely not strongly causally mediated. I'd be extremely cautious making any strong causal mediation claims through any of the layers. Importantly, I only ran the analysis on 50 pairs of examples, which is likely insufficient to give reliable analysis results. Given more time, it would be prudent to run this analysis on many more samples, and to make sure that whatever bug may be impacting benchmarking is not also impacting this mediation analysis.
- Results from the Mistral run can be found in the png and json files in the causal_mediation folder.
- My code has the infrastructure for position-sensitive patching, which would be especially useful for answering our question of whether we have a latent representation of a *running* count, but we're not using it in the main analysis. This would be room for great improvement in future iteration.
- For the mediation analysis, I sampled pairs of word lists randomly from my systematically generated data. Based on my validation results, I know this data is well balanced which should help with any confounding effects. However, there are certainly other, more sophisticated ways to control for confounders. One might try to either perturb the datum to create pairs or choose similar pairs (but then the signal may be too weak), use systematic variation to control for sepcific factors, etc. The key is that you want changes in count that aren't well correlated with other features.

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
