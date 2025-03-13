# Author: Thijs Brekhof
# Usage: Main classification file used for thesis - classifying if question asking / evidence presenting
# is identifyiable within a conversational turn
import argparse
import pandas as pd
import json
import argparse
import re
import string
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_rfc_data():
    unflattened_df = pd.read_json('rfc_predecessor_pairs_plain_text (4).json')
    #unflattened_df = pd.read_json('rfc_test.json')
    discussion_rfc_df = pd.json_normalize(unflattened_df['rfc_predecessor_pairs'])

    all_data = []
    # Process each discussion separately
    for idx, row in discussion_rfc_df.iterrows():
        # Process predecessor discussion
        if pd.notna(row['predecessor_full_text_until_rfc_started']):
            pred_turns = split_discussion(row['predecessor_full_text_until_rfc_started'])
            full_pred_discussion = row['predecessor_full_text_until_rfc_started']
            for turn in pred_turns:
                all_data.append({
                    'text': turn,
                    'full_discussion': full_pred_discussion,
                    'source_column': 'predecessor_full_text_until_rfc_started',
                    'original_index': idx,
                    'rfc_id': row['rfc_id'],
                    'predecessor_id': row['predecessor_id']
                })

        # Process RFC discussion
        if pd.notna(row['rfc_discussion_full_text']):
            rfc_turns = split_discussion(row['rfc_discussion_full_text'])
            full_rfc_discussion = row['predecessor_full_text_until_rfc_started']
            for turn in rfc_turns:
                all_data.append({
                    'text': turn,
                    'full_discussion': full_rfc_discussion,
                    'source_column': 'rfc_discussion_full_text',
                    'original_index': idx,
                    'rfc_id': row['rfc_id'],
                    'predecessor_id': row['predecessor_id']
                })

    # Create the DataFrame
    rfc_df = pd.DataFrame(all_data)

    return rfc_df


def split_discussion(text):
    # Pattern to match the end of a turn with two alternatives (with or without parentheses):
    # 1. [[User:Name|Name]] ([[User talk:Name|talk]])\n
    # 2. [[User:Name|Name]] [[User talk:Name|T]]\n
    turn_end_pattern = r'(?:(?:\[\[User:[^\]]+\]\]|<font[^>]*>[^<]*</font>).*?\[\[User[ _]talk:[^\]]+\]\](?:\)|[^\(\)]*?)\n)'

    # Find all indices where turns end
    turn_ends = []
    for match in re.finditer(turn_end_pattern, text):
        turn_ends.append(match.end())

    # Split the text into turns using these indices
    turns = []
    start_idx = 0

    for end_idx in turn_ends:
        turn = text[start_idx:end_idx]
        if turn.strip():
            turns.append(turn)
        start_idx = end_idx

    # Add the remaining text (if any) as the last turn
    if start_idx < len(text) and text[start_idx:].strip():
        turns.append(text[start_idx:])

    # Clean the turns
    cleaned_turns = []
    for turn in turns:
        if not turn.strip():
            continue

        # Clean the turn text while preserving structure
        turn_text = turn

        # Remove HTML tags but preserve content
        turn_text = re.sub(r'<[^>]+>', ' ', turn_text)

        # Preserve Wikipedia policy references
        wp_policies = re.findall(r'\[\[WP:[^\]]+\]\]|\[\[Wikipedia:[^\]]+\]\]|WP:[A-Z]+(?:/[A-Z]+)*', turn_text)
        for i, policy in enumerate(wp_policies):
            turn_text = turn_text.replace(policy, f"WPPOLICY{i}")

        # Clean up formatting markers while preserving structure
        turn_text = re.sub(r'^[\s\n]*[:*]+\s*', '', turn_text)
        turn_text = re.sub(r'\n[\s\n]*[:*]+\s*', '\n', turn_text)

        # Clean up spacing while preserving meaningful newlines
        turn_text = re.sub(r'\s*\n\s*', '\n', turn_text)
        turn_text = re.sub(r'[ \t]+', ' ', turn_text)
        turn_text = turn_text.strip()

        # Restore Wikipedia policy references
        for i, policy in enumerate(wp_policies):
            turn_text = turn_text.replace(f"WPPOLICY{i}", policy)

        if turn_text and not all(char in string.punctuation or char.isspace() for char in turn_text):
            cleaned_turns.append(turn_text)

    return cleaned_turns


def get_fact_check_interaction_data():
    """
    Process Wiki-Fact-check-Interaction data with full discussion context.
    """
    # Read the JSON file
    with open('sampled_interact_data.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    all_data = []

    # Iterate through each discussion
    for discussion in data:
        # Extract comments from the discussion
        comments = discussion['COMMENTS']

        # Create structured discussion text with proper indentation so the models get information
        # about which reply occurs where in a discussion
        structured_discussion = []
        for comment in comments:
            # Create indentation based on level in a format similar to how the actual discussions
            # are formatted on Wikipedia Talk Pages
            indent = "    " * comment['LEVEL']
            formatted_comment = f"{indent}{comment['TEXT-CLEAN']}"
            structured_discussion.append(formatted_comment)

        # Join all formatted comments with double newline for clarity
        full_discussion = "\n\n".join(structured_discussion)

        # Process each comment as a turn
        for comment in comments:
            all_data.append({
                'text': comment['TEXT-CLEAN'],
                'full_discussion': full_discussion,
                'discussion_id': discussion['DISCUSSION-ID'],
                'comment_id': comment['COMMENT-ID'],
                'timestamp': comment['TIMESTAMP'],
                'user': comment['USER']
            })

    # Create DataFrame from processed data
    df_fact_check = pd.DataFrame(all_data)

    return df_fact_check


def analyze_conversation_turn(turn_text, full_discussion, questions, model, tokenizer, args):
    """
    Analyze a single conversational turn within its discussion context to identify which questions it addresses.

    Args:
        turn_text (str): The specific conversational turn text to analyze
        full_discussion (str): The complete discussion text containing the turn
        questions (list): List of questions to check for
        model: The loaded model
        tokenizer: The tokenizer for the model
        args: the command line arguments such as model name
    Returns:
        dict: Dictionary mapping questions to boolean values
    """
    if args.use_full_context:
        context = full_discussion

        # Set maximum tokens for context since larger models have trouble with OOM errors
        if "medium" in args.model:
            max_context = 2048

            # Tokenize the context
            context_tokens = tokenizer.encode(context, add_special_tokens=False)

            # If context is too long, truncate from the beginning to maintain recent context
            if len(context_tokens) > max_context:
                context_tokens = context_tokens[-max_context:]
                # Decode back to text
                context = tokenizer.decode(context_tokens, skip_special_tokens=True)

        context_section = f"Full Discussion (only use this for context if you are unsure):\n{context}\n\n"
    else:
        context_section = ""

    questions_str = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])

    # Create prompt for the model
    prompt = f""" You are an expert analyst specializing in evaluating online discussions and source reliability 
    assessment. Your specific expertise includes: 1. Analyzing conversation patterns in collaborative environments 2. 
    Identifying explicit and implicit references to source reliability 3. Understanding the context and nuances of 
    Wikipedia-style discussions 4. Recognizing various aspects of source evaluation 

        Key Concepts:
    - A conversational turn is a single contribution by one participant in a larger discussion
    - Source reliability refers to the credibility, accuracy, and trustworthiness of information sources
    - Discussions may contain both explicit statements and implicit indicators about source reliability

    Task: Given a specific conversational turn from a discussion, you must determine which aspects of source reliability 
    are being addressed. For each of the 43 questions below, provide a YES/NO response indicating whether the turn 
    meaningfully addresses that aspect of source reliability. 

    {context_section}

    Conversational Turn: "{turn_text}"

    Questions to Identify:
    {questions_str}

    Instructions:
    1. For EACH question, carefully evaluate the conversational turn
    2. Determine if the turn meaningfully addresses the question's core concept
    3. Look for both explicit and implicit references
    4. Ignore superficial keyword mentions without substantive discussion

    Response Format:
    - Provide ONLY a YES or NO for EACH question
    - Do NOT provide any explanation for your analysis
    - Place your response on a new line, in the EXACT order of the questions
    - EXACTLY 43 numbered responses required
    - Each response MUST start with the question number
    - Example:
      1. YES
      2. NO
      3. YES
      4. NO
      ...
    Your comprehensive analysis: """

    # Debug prints
    # print(f"Processing question: {question}")
    # print(f"Input text: {turn_text[:20]}...")

    # To handle the tags, they are different depending on if you use phi or llama models
    prompt = get_model_specific_prompt(prompt, args.model)

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Get the length of input tokens to track where new tokens begin
    input_length = inputs.input_ids.shape[1]
    # Max length can be changed here

    if "llama" in args.model:
        outputs = model.generate(**inputs, max_new_tokens=args.output_length, pad_token_id=tokenizer.eos_token_id)
    else:
        outputs = model.generate(**inputs, max_new_tokens=args.output_length)

    # Only consider generated tokens when parsing
    new_tokens = outputs[0][input_length:]

    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Check response in case there is doubt about the output
    # print(f"generated tokens: {response}")
    # Split into lines and clean
    response_lines = [line.strip().lower() for line in response.split('\n')
                      if line.strip() and any(word in line.lower() for word in ['yes', 'no'])]
    results = {}
    # Process all questions at once
    for i, question in enumerate(questions):
        found_answer = False
        for line in response_lines:
            if str(i + 1) in line or question.lower() in line:
                results[question] = 'yes' if 'yes' in line.lower() else 'no'
                found_answer = True
                break
        if not found_answer:
            results[question] = 'no'

    return results


def process_dataset(data, questions, model, tokenizer, text_column, args, memory_manager, current_idx, batch_size=32):
    """
    Process entire dataset with improved memory management and proper checkpoint resuming.
    """

    memory_manager.clean_memory()

    max_retries = 3
    min_batch_size = 1

    # Create result columns if it doesn't yet exist
    for question in questions:
        question_col = f'addresses_{question.replace("?", "").replace(" ", "_")}'
        if question_col not in data.columns:
            data[question_col] = pd.NA

    total_rows = len(data)
    print(f"C. Processing {total_rows} total rows, total might be off if resuming from checkpoint")

    while current_idx < total_rows:
        retry_count = 0
        while retry_count < max_retries:
            try:
                end_idx = min(current_idx + batch_size, total_rows)
                batch = data.iloc[current_idx:end_idx]
                print(
                    f"Processing batch {current_idx // batch_size + 1} of {(total_rows + batch_size - 1) // batch_size}")

                # Single memory check using MemoryManager
                if memory_manager.is_memory_critical():
                    raise RuntimeError("Memory usage too high even after cleanup")

                # Process batch
                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = []
                    for idx, row in batch.iterrows():
                        # Check if this row has already been processed
                        sample_question = questions[0]
                        sample_col = f'addresses_{sample_question.replace("?", "").replace(" ", "_")}'

                        if pd.isna(data.at[idx, sample_col]):
                            future = executor.submit(
                                analyze_conversation_turn,
                                row[text_column],
                                row['full_discussion'],
                                questions,
                                model,
                                tokenizer,
                                args
                            )
                            futures.append((idx, future))

                    # Collect results as they complete
                    for idx, future in futures:
                        try:
                            results = future.result()
                            for question, is_addressed in results.items():
                                question_col = f'addresses_{question.replace("?", "").replace(" ", "_")}'
                                data.at[idx, question_col] = is_addressed
                        except Exception as e:
                            print(f"Error processing row {idx}: {e}")

                # Successful batch processing
                current_idx = end_idx

                # Save checkpoint if needed
                if args.checkpoint_freq > 0 and (current_idx % args.checkpoint_freq == 0):
                    save_checkpoint(data, args, current_idx)

                memory_manager.adjust_threshold(batch_success=True)
                break

            except RuntimeError as e:
                print(f"Batch processing failed (attempt {retry_count + 1}): {e}")
                retry_count += 1
                memory_manager.clean_memory()

                if batch_size > min_batch_size:
                    batch_size = max(min_batch_size, batch_size // 2)
                    print(f"Reduced batch size to {batch_size}")
                elif retry_count == max_retries:
                    print(f"Failed to process batch after {max_retries} attempts")
                    raise

    return data


def get_model_specific_prompt(prompt_text, model_name):
    """Add model-specific formatting to prompts"""
    if "Phi" in model_name:
        return f"<|user|>{prompt_text}<|end|><|assistant|>"
    elif "llama" in model_name.lower():
        return f"[INST] {prompt_text} [/INST]"
    else:
        return prompt_text


def setup_model(model_name):
    """
    Setup model with proper error handling and device checking.
    """
    print("Starting model setup...")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. This code requires GPU to run.")

    # Check number of available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")

    if "llama" in model_name.lower():
        print("Using LLama model: Logging in using HF token...")

        hf_token = os.getenv("HUGGING_FACE_TOKEN")
    else:
        hf_token = None
    print("Initializing tokenizer and model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        print("Tokenizer loaded successfully")

        # Using multiple gpus and divide load balanced amongst all available gpus
        if n_gpus > 1:
            device_map = "balanced"
            # 85% of total memory to avoid OOM errors
            gpu_memory = int(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2 * 0.85)
            max_memory = {i: f"{gpu_memory}MB" for i in range(n_gpus)}
        else:
            device_map = "auto"
            max_memory = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            force_download=True,
            token=hf_token,
            max_memory=max_memory
        )

        if n_gpus > 1:
            # Ensure model is properly distributed
            torch.cuda.synchronize()
            print("Synchronized model with multiple GPUs")

        print("Model loaded successfully")
        # Memory managar object to keep track of memory in case issues arise
        memory_manager = MemoryManager()

        return model, tokenizer, memory_manager

    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


class MemoryManager:
    def __init__(self, history_window=100):
        self.warning_threshold = 0.80
        self.critical_threshold = 0.90
        self.history_window = history_window
        self.history = []
        self.threshold = 0.85
        self.threshold_adjustment_rate = 0.05

    def clean_memory(self):
        """Centralized memory cleanup"""
        torch.cuda.empty_cache()
        gc.collect()

    def check_memory(self):
        """Check memory status and handle cleanup if needed"""
        if not torch.cuda.is_available():
            return None

        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2

        stats = {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'total_mb': total,
            'available_mb': total - allocated,
            'utilization_percent': (allocated / total) * 100
        }

        self.history.append(stats['allocated_mb'])
        if len(self.history) > self.history_window:
            self.history.pop(0)

        # Handle memory thresholds
        if stats['utilization_percent'] > self.critical_threshold * 100:
            print("Critical: Memory usage too high, forcing cleanup")
            self.clean_memory()
            # Recheck memory after cleanup
            return self.check_memory()
        elif stats['utilization_percent'] > self.warning_threshold * 100:
            print(f"Warning: Memory utilization at {stats['utilization_percent']:.2f}%")

        return stats

    def is_memory_critical(self):
        """Check if memory usage is at critical levels"""
        stats = self.check_memory()
        return stats['utilization_percent'] > self.critical_threshold * 100

    def adjust_threshold(self, batch_success=True):
        """
        Adjust memory threshold based on processing success
        Args:
            batch_success (bool): Whether the last batch was processed successfully
        """
        if batch_success:
            # If successful, we can try to be more aggressive (increase threshold)
            self.threshold = min(0.95, self.threshold + self.threshold_adjustment_rate)
        else:
            # If failed, we need to be more conservative (decrease threshold)
            self.threshold = max(0.70, self.threshold - self.threshold_adjustment_rate)


def save_checkpoint(data, args, current_index):
    # Create a copy and drop full_discussion column to reduce file size
    checkpoint_data = data.iloc[:current_index].copy()
    # To save file size
    checkpoint_data = checkpoint_data.drop(columns=['full_discussion'])

    checkpoint_file = os.path.join(
        args.output_dir,
        f"checkpoint_{args.data_source}_{current_index}.csv"
    )
    checkpoint_data.to_csv(checkpoint_file, index=False)
    print(f"Checkpoint saved: {checkpoint_file}")


def main():
    # Create parser
    parser = argparse.ArgumentParser(description='Process different types of wiki data')

    # Add argument for data type
    parser.add_argument('-ds', "--data_source", required=True, type=str,
                        choices=['rfc', 'interact'],
                        help='Type of data to process: rfc, or interact')
    parser.add_argument('-m', "--model", required=True,
                        choices=["microsoft/Phi-3.5-mini-instruct", "meta-llama/Llama-3.1-8B-Instruct",
                                 "microsoft/Phi-3-medium-128k-instruct"], type=str,
                        help='Name of model to use for inference: microsoft/Phi-3.5-mini-instruct, '
                             'meta-llama/Llama-3.1-8B-Instruct, or "microsoft/Phi-3-medium-128k-instruct')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=1000,
                        help='Frequency of checkpoints (in number of rows)')
    parser.add_argument('-res', '--resume_from', type=str,
                        help='Resume from checkpoint file')
    parser.add_argument('-out', '--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--output_length', '-len', type=int, default=10, help="The maximum length of newly generated "
                                                                              "tokens")
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of parallel workers for processing')
    parser.add_argument('--use_full_context', action='store_true',
                        help='Whether to use full discussion context or just turns')

    args = parser.parse_args()

    # Loading list of questions
    questions = []
    with open("prompt_questions.txt") as f:
        for question in f.readlines():
            questions.append(question.rstrip())

    # RFC data, conversational turns are under 'text'
    if args.data_source == 'rfc':
        print("Processing RFC data...")
        text_column = "text"
        data = get_rfc_data()

        print()

    # wiki fact check interaction, conversational turns are under 'text'
    elif args.data_source == 'interact':
        print("Processing fact check interaction data...")
        text_column = "text"
        data = get_fact_check_interaction_data()

    # If resuming from checkpoint:
    current_idx = 0
    if args.resume_from and os.path.exists(args.resume_from):
        try:
            # Load the checkpoint data
            checkpoint_data = pd.read_csv(args.resume_from)
            # Merge checkpoint data with original data
            data = data.combine_first(checkpoint_data)
            last_processed = int(args.resume_from.split('_')[-1].split('.')[0])
            current_idx = last_processed
            print(f"Resuming processing from index {current_idx}")
        except (ValueError, IndexError):
            print("Warning: Could not determine last processed index from checkpoint filename")
            exit(-1)


    # Loading model
    print(f"Loading model: {args.model}")
    model, tokenizer, memory_manager = setup_model(args.model)

    initial_stats = memory_manager.check_memory()
    if initial_stats:
        print(f"Initial memory usage: {initial_stats['utilization_percent']:.2f}%")
    print("10. Starting dataset processing...")
    # Creating directory before calling process_dataset as otherwise we run into problems when saving checkpoint
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Process the dataset with memory monitoring
        processed_data = process_dataset(
            data=data,
            questions=questions,
            model=model,
            tokenizer=tokenizer,
            text_column=text_column,
            args=args,
            memory_manager=memory_manager,
            batch_size=args.batch_size,
            current_idx=current_idx
        )
        print("11. Dataset processing complete")

        # Remove the full_discussion column before saving
        processed_data = processed_data.drop(columns=['full_discussion'])

        # Create suffix based on model name and context setting
        # E.g. 'Phi-3.5-mini-instruct' from full path
        model_name = args.model.split('/')[-1]
        context_type = "full" if args.use_full_context else "turn"

        # Save results

        output_file = os.path.join(
            args.output_dir,
            f"results_rfc_{model_name}_{context_type}.csv"
        )
        processed_data.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

    finally:
        # Some stats about memory usage to keep track of resources
        if torch.cuda.is_available():
            final_stats = memory_manager.check_memory()
            if final_stats and len(memory_manager.history) > 0:
                print("\nMemory Usage Summary:")
                print(f"Peak memory usage: {max(memory_manager.history):.2f} MB")
                print(f"Average memory usage: {sum(memory_manager.history) / len(memory_manager.history):.2f} MB")
                print(f"Final memory usage: {final_stats['utilization_percent']:.2f}%")

                # Calculate memory efficiency
                memory_efficiency = (sum(memory_manager.history) / len(memory_manager.history)) / final_stats[
                    'total_mb']
                print(f"Overall memory efficiency: {memory_efficiency:.2%}")


if __name__ == "__main__":
    main()
