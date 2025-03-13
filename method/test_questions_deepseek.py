# Author: Thijs Brekhof
# Usage: test_questions.py but for deepseek specifically

import argparse
import pandas as pd
from openai import OpenAI
import os
import json
import time
from test_questions import get_rfc_data, get_fact_check_interaction_data, split_discussion

def setup_client():
    """Setup the DeepSeek API client"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def retry_api_call(client, prompt, max_retries=3):
    """Retry API call with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=270
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            wait_time = (2 ** attempt) * 1
            print(f"Attempt {attempt + 1} failed. Waiting {wait_time} seconds...")
            time.sleep(wait_time)


def analyze_turn(turn_text, full_discussion, questions, client, args):
    """Analyze a single turn using DeepSeek"""
    if args.use_full_context:
        context_section = f"Full Discussion (only use this for context if you are unsure):\n{full_discussion}\n\n"
    else:
        context_section = ""

    questions_str = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])

    prompt = f"""You are an expert analyst specializing in evaluating online discussions and source reliability 
        assessment. Your specific expertise includes: 1. Analyzing conversation patterns in collaborative environments 2. 
        Identifying explicit and implicit references to source reliability 3. Understanding the context and nuances of 
        Wikipedia-style discussions 4. Recognizing various aspects of source evaluation 

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
        - Place your response on a new line, in the EXACT order of the questions
        - EXACTLY 43 numbered responses required
        - Each response MUST start with the question number
        Example:
        1. YES
        2. NO
        etc."""

    response_text = retry_api_call(client, prompt)

    if response_text is None:
        return {question: 'ERROR' for question in questions}

    response_lines = [line.strip().lower() for line in response_text.split('\n')
                      if line.strip() and any(word in line.lower() for word in ['yes', 'no'])]

    results = {}
    for i, question in enumerate(questions):
        found_answer = False
        for line in response_lines:
            if str(i + 1) in line or question.lower() in line:
                results[question] = 'yes' if 'yes' in line.lower() else 'no'
                found_answer = True
                break
        if not found_answer:
            results[question] = 'ERROR'

    return results


def save_checkpoint(processed_data, args, current_index):
    """Save current progress to checkpoint file with row count in filename"""
    # Create a copy and drop full_discussion column to reduce file size
    checkpoint_data = processed_data.copy()
    if 'full_discussion' in checkpoint_data.columns:
        checkpoint_data = checkpoint_data.drop(columns=['full_discussion'])

    checkpoint_file = os.path.join(
        "results",
        args.data_source,
        "deepseek_turn",
        f"checkpoint_{args.data_source}_{current_index}.csv"
    )
    checkpoint_data.to_csv(checkpoint_file, index=False)
    print(f"Checkpoint saved to {checkpoint_file} ({current_index} rows processed)")


def load_checkpoint(checkpoint_file, original_data):
    """Load progress from checkpoint file"""
    if os.path.exists(checkpoint_file):
        checkpoint_data = pd.read_csv(checkpoint_file)
        processed_indices = checkpoint_data.index
        remaining_data = original_data.drop(processed_indices)
        return checkpoint_data, remaining_data
    return pd.DataFrame(), original_data


def main():
    parser = argparse.ArgumentParser(description='Process discussions with DeepSeek')
    parser.add_argument('-ds', '--data_source', required=True,
                        choices=['rfc', 'interact'],
                        help='Type of data to process')
    parser.add_argument('--use_full_context', action='store_true',
                        help='Whether to use full discussion context')
    parser.add_argument('--checkpoint_freq', type=int, default=500,
                        help='Save checkpoint every N turns')
    parser.add_argument('--resume_from', type=str,
                        help='Resume from specific checkpoint file')
    parser.add_argument('--retry_errors', type=str,
                        help='Retry processing errors from specific output file')

    args = parser.parse_args()

    # Setup output directory structure
    output_dir = f"results/{args.data_source}/deepseek_turn"
    os.makedirs(output_dir, exist_ok=True)

    # Setup DeepSeek client
    client = setup_client()

    # Load questions
    with open("prompt_questions.txt") as f:
        questions = [question.rstrip() for question in f.readlines()]

    if args.retry_errors:
        # Load previous results and retry error cases
        previous_results = pd.read_csv(args.retry_errors)
        error_mask = previous_results[questions].eq('ERROR').any(axis=1)
        data_to_process = previous_results[error_mask]
        existing_results = previous_results[~error_mask]
    else:
        # Load original data based on data source
        if args.data_source == 'rfc':
            data_to_process = get_rfc_data()
        elif args.data_source == 'interact':
            data_to_process = get_fact_check_interaction_data()
        existing_results = pd.DataFrame()

    # Setup checkpoint filename
    checkpoint_file = os.path.join(output_dir, f"checkpoint_{args.data_source}.csv")

    # Load from checkpoint if specified
    if args.resume_from:
        existing_results, data_to_process = load_checkpoint(args.resume_from, data_to_process)

    # Process each turn
    results = []
    for idx, row in data_to_process.iterrows():
        print(f"Processing turn {idx + 1}/{len(data_to_process)}")

        # Add delay to respect rate limits and avoid random crashes
        time.sleep(1)

        # Analyze turn
        model_results = analyze_turn(
            row['text'],
            row.get('full_discussion', ''),
            questions,
            client,
            args
        )

        # Create result row
        result_row = row.copy()
        for question in questions:
            result_row[question] = model_results[question]
        results.append(result_row)

        # Save checkpoint if needed
        if (idx + 1) % args.checkpoint_freq == 0:
            current_results = pd.concat([existing_results, pd.DataFrame(results)], ignore_index=True)
            current_index = len(current_results)  # Get the current count of processed rows
            save_checkpoint(current_results, args, current_index)

    # Combine all results
    final_results = pd.concat([existing_results, pd.DataFrame(results)], ignore_index=True)

    # Save final results
    context_type = "full" if args.use_full_context else "turn"
    output_file = os.path.join(
        output_dir,
        f"results_deepseek_{args.data_source}_{context_type}.csv"
    )
    final_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()