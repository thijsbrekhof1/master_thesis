# Author: Thijs Brekhof
# Usage: test_questions.py but for claude models specifically

import pandas as pd
from anthropic import Anthropic
import os
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
# Reuse existing data loading functions
from test_questions import get_rfc_data, get_fact_check_interaction_data, save_checkpoint


def load_questions():
    """Load questions from file"""
    with open("prompt_questions.txt") as f:
        return [q.rstrip() for q in f.readlines()]


def get_group_id(row):
    return (row['predecessor_id'] if row['source_column'] == 'predecessor_full_text_until_rfc_started'
            else row['rfc_id'])


class ClaudeAnalyzer:
    def __init__(self, model="claude-3-5-sonnet-20241022"):
        self.client = Anthropic(
            api_key="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")
        self.model = model
        self.questions = load_questions()
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self):
        """Create system prompt with all static content"""
        questions_str = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(self.questions)])

        return f"""You are an expert analyst specializing in evaluating online discussions and source reliability 
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
          ..."""

    def analyze_turn(self, turn_text, full_discussion=None, use_full_context=False):
        """Analyze a single conversational turn"""
        system_content = [{
            "type": "text",
            "text": self.system_prompt,
            "cache_control": {"type": "ephemeral"}
        }]

        message_content = []
        if use_full_context and full_discussion:
            message_content.append({
                "type": "text",
                "text": f"Full Discussion Context:\n{full_discussion}",
                "cache_control": {"type": "ephemeral"}})

        message_content.append({
            "type": "text",
            "text": f"Conversational Turn: \"{turn_text}\""})

        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_content,
                messages=[{
                    "role": "user",
                    "content": message_content
                }],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                max_tokens=1024
            )

            # Process the response
            response_lines = [line.strip().lower() for line in response.content[0].text.split('\n')
                              if line.strip() and any(word in line.lower() for word in ['yes', 'no'])]

            results = {}
            for i, question in enumerate(self.questions):
                found_answer = False
                for line in response_lines:
                    if str(i + 1) in line:
                        results[question] = 'yes' if 'yes' in line.lower() else 'no'
                        found_answer = True
                        break
                if not found_answer:
                    results[question] = 'no'

            return results

        except Exception as e:
            print(f"Error in analyze_turn: {e}")
            # Return default 'no' responses in case of error
            return {question: 'no' for question in self.questions}

    def process_dataset(self, data, text_column, args, use_full_context=False):
        """Process entire dataset with optimized caching per discussion"""
        # Initialize result columns if they don't exist (based on questions)
        for question in self.questions:
            question_col = f'addresses_{question.replace("?", "").replace(" ", "_")}'
            if question_col not in data.columns:
                data[question_col] = pd.NA

        # Initialize current_idx for resuming
        current_idx = 0
        total_rows = len(data)
        if args.resume_from and os.path.exists(args.resume_from):
            try:
                # Load checkpoint data
                checkpoint_data = pd.read_csv(args.resume_from)
                # Update data with processed results
                for question in self.questions:
                    question_col = f'addresses_{question.replace("?", "").replace(" ", "_")}'
                    if question_col in checkpoint_data.columns:
                        data[question_col].update(checkpoint_data[question_col])
                # Extract last processed index from filename
                last_processed = int(args.resume_from.split('_')[-1].split('.')[0])
                current_idx = last_processed
                print(f"Resuming processing from index {current_idx}")
                print(f"Processing {total_rows - current_idx} total (new) rows")

            # Kill process if checkpoint cant be found to avoid unnecessary api costs
            except (ValueError, IndexError):
                print("Warning: Could not determine last processed index from checkpoint filename")
                print("Terminating process.")
                exit(-1)

        else:
            print(f"Processing {total_rows} total rows")

        # Process rows based on data source
        # interact data
        if 'discussion_id' in data.columns:
            # For interaction data
            current_discussion_id = None
            current_discussion = None
            for idx, row in data.iloc[current_idx:].iterrows():
                if row['discussion_id'] != current_discussion_id:
                    # New discussion started
                    current_discussion_id = row['discussion_id']
                    current_discussion = row['full_discussion']
                    print(f"Processing new discussion: {current_discussion_id}")

                # Check if this row needs processing
                sample_question = self.questions[0]
                sample_col = f'addresses_{sample_question.replace("?", "").replace(" ", "_")}'

                if pd.isna(data.at[idx, sample_col]):
                    results = self.analyze_turn(row[text_column],
                                                current_discussion if use_full_context else None, use_full_context)
                    # Store results
                    for question, is_addressed in results.items():
                        question_col = f'addresses_{question.replace("?", "").replace(" ", "_")}'
                        data.at[idx, question_col] = is_addressed

                # Save checkpoint if needed
                if args.checkpoint_freq > 0 and (idx + 1) % args.checkpoint_freq == 0:
                    save_checkpoint(data, args, idx + 1)

        # RFC data
        elif 'source_column' in data.columns:
            # For RFC data
            current_rfc_id = None
            current_predecessor_id = None
            current_discussion = None

            for idx, row in data.iloc[current_idx:].iterrows():
                # For RFC discussions
                if row['source_column'] == 'rfc_discussion_full_text':
                    if row['rfc_id'] != current_rfc_id:
                        # New RFC discussion started
                        current_rfc_id = row['rfc_id']
                        current_discussion = row['full_discussion']
                        print(f"Processing new RFC discussion: {current_rfc_id}")

                # Predecessor discussions
                else:
                    if row['predecessor_id'] != current_predecessor_id:
                        # New predecessor discussion started
                        current_predecessor_id = row['predecessor_id']
                        current_discussion = row['full_discussion']
                        print(f"Processing new predecessor discussion: {current_predecessor_id}")

                # Check if this row needs processing
                sample_question = self.questions[0]
                sample_col = f'addresses_{sample_question.replace("?", "").replace(" ", "_")}'

                if pd.isna(data.at[idx, sample_col]):
                    results = self.analyze_turn(row[text_column],
                                                current_discussion if use_full_context else None, use_full_context)
                    # Store results
                    for question, is_addressed in results.items():
                        question_col = f'addresses_{question.replace("?", "").replace(" ", "_")}'
                        data.at[idx, question_col] = is_addressed

                # Save checkpoint if needed
                if args.checkpoint_freq > 0 and (idx + 1) % args.checkpoint_freq == 0:
                    save_checkpoint(data, args, idx + 1)

            return data


def main():
    parser = argparse.ArgumentParser(description='Process wiki data using Claude-3-Sonnet')
    parser.add_argument('--data_source', required=True, choices=['rfc', 'interact'],
                        help='Type of data to process')
    parser.add_argument('--use_full_context', action='store_true',
                        help='Whether to use full discussion context', default=False)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing')
    parser.add_argument('--output_dir', type=str, default='results/claude/',
                        help='Directory to save results')
    parser.add_argument('--resume_from', type=str,
                        help='Resume from checkpoint file')
    parser.add_argument('--checkpoint_freq', type=int, default=999,
                        help='Frequency of checkpoints (in number of rows)')

    args = parser.parse_args()

    # Do this here to avoid errors when trying to save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = ClaudeAnalyzer()

    # Load appropriate dataset
    if args.data_source == 'rfc':
        print("Processing RFC data...")
        data = get_rfc_data()
        text_column = 'text'

    elif args.data_source == "interact":
        print("Processing fact check interaction data...")
        data = get_fact_check_interaction_data()
        text_column = 'text'

    # Process dataset
    processed_data = analyzer.process_dataset(
        data=data,
        text_column=text_column,
        use_full_context=args.use_full_context,
        args=args
    )

    # Save results

    context_type = "full" if args.use_full_context else "turn"
    output_file = os.path.join(
        args.output_dir,
        f"results_{args.data_source}_claude_sonnet3.5_{context_type}.csv"
    )

    # Remove full_discussion column before saving
    processed_data = processed_data.drop(columns=['full_discussion'])
    processed_data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
