# api key =
# Author: Thijs Brekhof
# Usage: to run the Deep Seek model on the data samples included in the manually annotated subset

import argparse
import pandas as pd
from openai import OpenAI
import os
import time



def setup_client():
    """Setup the DeepSeek API client"""
    api_key = ""

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
    """Analyze a single turn using DeepSeek Reasoner"""
    if args.use_full_context:
        context_section = f"Full Discussion (only use this for context if you are unsure):\n{full_discussion}\n\n"
    else:
        context_section = ""

    questions_str = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])

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

    response_text = retry_api_call(client, prompt)

    if response_text is None:
        return {question: 'ERROR' for question in questions}
    else:
        print(response_text)

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


def save_checkpoint(processed_data, checkpoint_file):
    """Save current progress to checkpoint file"""
    processed_data.to_csv(checkpoint_file, index=False)
    print(f"Checkpoint saved to {checkpoint_file}")


def load_checkpoint(checkpoint_file, original_data):
    """Load progress from checkpoint file"""
    if os.path.exists(checkpoint_file):
        checkpoint_data = pd.read_csv(checkpoint_file)
        # Get indices of processed turns
        processed_indices = checkpoint_data.index
        # Get remaining turns to process
        remaining_data = original_data.drop(processed_indices)
        return checkpoint_data, remaining_data
    return pd.DataFrame(), original_data


def main():
    parser = argparse.ArgumentParser(description='Process manual annotation results with DeepSeek Reasoner')
    parser.add_argument('--use_full_context', action='store_true',
                        help='Whether to use full discussion context or just turns')

    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Save checkpoint every N turns')
    parser.add_argument('--resume_from_checkpoint', type=str,
                        help='Resume from specific checkpoint file')
    parser.add_argument('--retry_errors', type=str,
                        help='Retry processing errors from specific output file')

    args = parser.parse_args()

    output_dir = "results/manual_annotation/deepseek"
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup DeepSeek client
    client = setup_client()

    # Loading list of questions
    with open("prompt_questions.txt") as f:
        questions = [question.rstrip() for question in f.readlines()]

    if args.retry_errors:
        # Load previous results and retry error cases
        previous_results = pd.read_csv(args.retry_errors)
        # Filter rows where any question has 'ERROR'
        error_mask = previous_results[questions].eq('ERROR').any(axis=1)
        data_to_process = previous_results[error_mask]
        existing_results = previous_results[~error_mask]
    else:
        # Load original data
        data_to_process = pd.read_csv('results/manual_annotation/Manual_Annotation_results.csv')
        # Remove any existing annotation columns
        data_to_process = data_to_process.drop(columns=questions, errors='ignore')
        existing_results = pd.DataFrame()

    # Setup checkpoint filename

    checkpoint_file = os.path.join(output_dir, f"checkpoint.csv")

    # Load from checkpoint if specified
    if args.resume_from_checkpoint:
        existing_results, data_to_process = load_checkpoint(args.resume_from_checkpoint, data_to_process)

    # Process each turn
    results = []
    for idx, row in data_to_process.iterrows():
        print(f"Processing turn {idx + 1}/{len(data_to_process)}")

        # Add delay to respect rate limits
        time.sleep(1)

        # Analyze turn
        model_results = analyze_turn(
            row['Turn'],
            row['Discussion text'] if args.use_full_context else "",
            questions,
            client,
            args
        )

        # Create result row with all original columns
        result_row = row.copy()
        # Update the question columns with model results
        for question in questions:
            result_row[question] = model_results[question]
        results.append(result_row)

        # Save checkpoint if needed
        if (idx + 1) % args.checkpoint_interval == 0:
            current_results = pd.concat([existing_results, pd.DataFrame(results)], ignore_index=True)
            save_checkpoint(current_results, checkpoint_file)

    # Combine all results
    final_results = pd.concat([existing_results, pd.DataFrame(results)], ignore_index=True)

    # Save final results
    context_type = "full" if args.use_full_context else "turn"
    output_file = os.path.join(
        output_dir,
        f"results_deepseek_chat_{context_type}.csv"
    )
    final_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()