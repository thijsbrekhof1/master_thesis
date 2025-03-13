# Author: Thijs Brekhof
# Usage: To classify the specific turns which were manually annotated (csv file format)

import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def setup_model(model_name):
    """Same as in current_code"""
    print("Starting model setup...")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. This code requires GPU to run.")

    if "llama" in model_name.lower():
        print("Using LLama model: Logging in using HF token...")
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
    else:
        hf_token = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            token=hf_token
        )
        return model, tokenizer

    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def get_model_specific_prompt(prompt_text, model_name):
    """Same as in current_code"""
    if "Phi" in model_name:
        return f"<|user|>{prompt_text}<|end|><|assistant|>"
    elif "llama" in model_name.lower():
        return f"[INST] {prompt_text} [/INST]"
    else:
        return prompt_text


def analyze_turn(turn_text, full_discussion, questions, model, tokenizer, args):
    """Similar to analyze_conversation_turn but simplified"""
    if args.use_full_context:
        context_section = f"Full Discussion (only use this for context if you are unsure):\n{full_discussion}\n\n"
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

    prompt = get_model_specific_prompt(prompt, args.model)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    if "llama" in args.model:
        outputs = model.generate(**inputs, max_new_tokens=args.output_length, pad_token_id=tokenizer.eos_token_id)
    else:
        outputs = model.generate(**inputs, max_new_tokens=args.output_length)

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    response_lines = [line.strip().lower() for line in response.split('\n')
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
            results[question] = 'no'

    return results


def main():
    parser = argparse.ArgumentParser(description='Process manual annotation results with LLM')
    parser.add_argument('-m', "--model", required=True,
                        choices=["microsoft/Phi-3.5-mini-instruct",
                                 "meta-llama/Llama-3.1-8B-Instruct",
                                 "microsoft/Phi-3-medium-128k-instruct"],
                        help='Model to use for inference')
    parser.add_argument('--use_full_context', action='store_true',
                        help='Whether to use full discussion context or just turns')
    parser.add_argument('-out', '--output_dir', type=str, default='results/manual',
                        help='Directory to save results')
    parser.add_argument('--output_length', '-len', type=int, default=10,
                        help='Maximum length of newly generated tokens')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = pd.read_csv('Manual_Annotation_results.csv')

    # Loading list of questions
    questions = []
    with open("prompt_questions.txt") as f:
        for question in f.readlines():
            questions.append(question.rstrip())

    # Setup model
    model, tokenizer = setup_model(args.model)

    # Process each turn
    results = []
    for idx, row in data.iterrows():
        print(f"Processing turn {idx + 1}/{len(data)}")

        # Analyze turn
        model_results = analyze_turn(
            row['Turn'],
            row['Discussion text'] if args.use_full_context else "",
            questions,
            model,
            tokenizer,
            args
        )

        # Create result row with only non-question columns
        result_row = row[['Dataset', 'Turn', 'Discussion text', "Discussion ID"]].copy()
        # Add model predictions with clear prefix
        for question in questions:
            result_row[f'model_{question}'] = model_results[question]

        results.append(result_row)

    # Create output dataframe
    output_df = pd.DataFrame(results)

    # Save results
    model_name = args.model.split('/')[-1]
    context_type = "full" if args.use_full_context else "turn"
    output_file = os.path.join(
        args.output_dir,
        f"results_{model_name}_{context_type}.csv"
    )

    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
