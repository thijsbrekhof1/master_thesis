## Analyzing the Fact-Checking Process in Wikipedia Deliberations

## Abstract

This thesis investigates the deliberation process on Wikipedia Talk pages, with a specific focus on fact-checking and source reliability verification. Wikipedia's collaborative knowledge platform relies heavily on editors' ability to evaluate sources and verify factual claims, though little research has systematically analyzed the patterns that characterize successful versus unsuccessful fact-checking deliberations. Sourcing from established argumentation theories such as Walton's argumentation schemes, Relevance Theory, Pragma-Dialectics, and Toulmin's model, we developed a comprehensive framework consisting of seven question categories and 43 specific questions to identify elements of the fact-checking deliberation process. We examine two distinct datasets: the RFC-Deliberation dataset, which pairs successful deliberations with their unsuccessful predecessors, and the Wiki-Fact-check-Interaction dataset, specifically constructed to analyze fact-checking discussions. Using prompt engineering techniques with various language models (Phi-3.5-mini, Phi-3-medium, Llama-3.1 and DeepSeek), we implement a zero-shot classification approach to automatically identify these elements within conversational turns across two settings. While our manual annotation process achieved high inter-annotator agreement (Cohen's Kappa of 0.918), automated classification proved challenging for all models, with DeepSeek achieving the best but still modest performance. Our exploratory sequential flow analysis reveals potentially meaningful differences between successful and unsuccessful deliberations, including longer and more complex argumentative structures in successful discussions, greater emphasis on relevance and source credibility, and less focus on conflict management. These findings, while not definitive due to the classification limitations, provide insights into the nature of effective fact-checking deliberations on Wikipedia and suggest areas for future research regarding fact-checking deliberations.


## Core Classification Tools

### test_questions.py

Primary tool for classifying conversational turns with LLMs.

python test_questions.py -ds [rfc|interact] -m [model_name] [options]

Options:
  -ds, --data_source      Type of data to process: rfc or interact
  -m, --model             Model to use: microsoft/Phi-3.5-mini-instruct, 
                          meta-llama/Llama-3.1-8B-Instruct, or microsoft/Phi-3-medium-128k-instruct
  -bs, --batch_size       Batch size for processing (default: 32)
  -cf, --checkpoint_freq  Checkpoint frequency in number of rows (default: 1000)
  -res, --resume_from     Resume from specific checkpoint file
  -out, --output_dir      Directory to save results (default: results)
  --output_length         Maximum length of newly generated tokens (default: 10)
  --num_workers           Number of parallel workers (default: 2)
  --use_full_context      Whether to use full discussion context

### test_questions_deepseek.py
Specialized version for the DeepSeek API.

python test_questions_deepseek.py -ds [rfc|interact] [options]

Options:
  -ds, --data_source     Type of data to process
  --use_full_context     Whether to use full discussion context
  --checkpoint_freq      Save checkpoint every N turns (default: 500)
  --resume_from          Resume from specific checkpoint file
  --retry_errors         Retry processing errors from specific output file


### test_questions_claude.py
Version that uses Claude 3.5 Sonnet API. NOTE: As of 02/25 claude sonnet 3.5 is replaced by claude sonnet 3.7

python test_questions_claude.py --data_source [rfc|interact] [options]

Options:
  --data_source          Type of data to process: rfc or interact
  --use_full_context     Whether to use full discussion context (default: False)
  --batch_size           Batch size for processing (default: 64)
  --max_workers          Maximum number of parallel workers (default: 2)
  --output_dir           Directory to save results (default: results/claude/)

## Manual Annotation Tools

### test_questions_manual.py
Processes manually annotated samples with LLM classification.

python test_questions_manual.py -m [model_name] [options]

Options:
  -m, --model            Model to use: microsoft/Phi-3.5-mini-instruct, 
                         meta-llama/Llama-3.1-8B-Instruct, or microsoft/Phi-3-medium-128k-instruct
  --use_full_context     Whether to use full discussion context
  -out, --output_dir     Directory to save results (default: results/manual)
  --output_length        Maximum length of newly generated tokens (default: 10)

  
### test_questions_deepseek_manual.py
DeepSeek API version for manual annotations.

python test_questions_deepseek_manual.py [options]

Options:
  --use_full_context         Whether to use full discussion context
  --checkpoint_interval      Save checkpoint every N turns (default: 10)
  --resume_from_checkpoint   Resume from specific checkpoint file
  --retry_errors             Retry processing errors from specific output file

## Data Preparation and Analysis Tools
### random_samples.txt
Tool to extract random samples from Wiki-Fact-check-Interaction and RFC datasets.

python random_samples.py

### subsample_data.py
Extracts discussions matching a list of sampled IDs.

python subsample_data.py

### cluster_discussions.py
Clusters discussions based on their summaries for more diverse sampling.

python cluster_summaries.py



## Evaluation and Analysis Tools
### compare_manual_annotation.py
Computes inter-annotator agreement metrics between two annotation sets.
python compare_manual_annotation.py

### evaluate_classification.py
Compares model predictions with manual annotations.
python evaluate_classification.py

### analyze_flows.py
Analyzes question patterns in discussions, extracting category flows.
python analyze_flows.py

### data_analysis.py
Comprehensive statistics about the datasets.
python data_analysis.py


## Setup Requirements

Python 3.8+
GPU environment for transformer-based models
API keys for DeepSeek and/or Claude APIs when using those models
HF token for LLama models
Dependencies in requirements.txt


## Dataset Preparation
The code expects these datasets:

Wiki-Fact-check-Interaction.jsonl: Wikipedia fact-checking interactions
rfc_predecessor_pairs_plain_text (4).json: RFC discussions with predecessor data

## Notes
Most scripts are designed to run on a computer cluster using shell scripts
Checkpointing is implemented to allow resuming long-running processes
Memory management is built in to handle GPU resource constraints

For additional information about each script, refer to the documentation at the beginning of each file and the comments/docstrings provided throughout.



  
