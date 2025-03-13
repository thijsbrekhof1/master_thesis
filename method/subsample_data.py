# Author: Thijs Brekhof
# Usage: subsample the fact-check-interaction dataset based on the IDs we gathered in our subsampling method

import pandas as pd
import json
from pathlib import Path


def load_sampled_ids(csv_path):
    """
    Load the discussion IDs from the sampled_discussions.csv file
    """
    df = pd.read_csv(csv_path)
    return set(df['discussion_id'])


def process_interact_data(input_jsonl_path, output_jsonl_path, sampled_ids):
    """
    Process the interaction dataset and extract discussions matching sampled IDs
    """
    count = 0

    # Create output directory if it doesn't exist
    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    # Process input file line by line to handle large files efficiently
    with open(input_jsonl_path, 'r', encoding='utf-8') as f_in, \
            open(output_jsonl_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            try:
                discussion = json.loads(line.strip())
                discussion_id = discussion['DISCUSSION-ID']

                # If this discussion is in sampled set, write it to output
                if discussion_id in sampled_ids:
                    json.dump(discussion, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    count += 1

                    # Print progress
                    if count % 100 == 0:
                        print(f"Processed {count} matching discussions")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in discussion data: {e}")
                continue

    return count


def verify_output(output_path, expected_ids):
    """
    Verify that the output file contains all expected discussions
    """
    found_ids = set()
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            discussion = json.loads(line)
            found_ids.add(discussion['DISCUSSION-ID'])

    missing_ids = expected_ids - found_ids
    if missing_ids:
        print(f"\nWarning: {len(missing_ids)} discussions were not found:")
        print(f"Missing IDs: {sorted(list(missing_ids))[:5]}...")
    else:
        print("\nVerification successful: All sampled discussions were extracted")


def main():
    # Configuration
    sampled_csv_path = 'results/sampling-method/sampled_discussions.csv'
    interact_data_path = '../data/Wiki-Fact-check-Interaction/Wiki-Fact-check-Interaction.jsonl'
    output_path = 'results/sampling-method/sampled_interact_data.jsonl'

    print("Loading sampled discussion IDs...")
    sampled_ids = load_sampled_ids(sampled_csv_path)
    print(f"Loaded {len(sampled_ids)} discussion IDs")

    print("\nProcessing interaction dataset...")
    processed_count = process_interact_data(
        input_jsonl_path=interact_data_path,
        output_jsonl_path=output_path,
        sampled_ids=sampled_ids
    )

    print(f"\nCompleted processing:")
    print(f"- Found and extracted {processed_count} discussions")
    print(f"- Output saved to: {output_path}")

    # Verify output
    verify_output(output_path, sampled_ids)




if __name__ == "__main__":
    main()