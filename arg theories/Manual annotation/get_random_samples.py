# Author: Thijs Brekhof
# Usage: extract random data points from both datasets to comprise the manual annotation subset

import random
import pandas as pd
import regex as re
import string


def wiki_interact_sample():
    df = pd.read_json('../../method/results/sampling-method/sampled_interact_data.jsonl', lines=True)

    # Randomly select 10 discussions
    selected_indices = random.sample(range(len(df)), 10)
    sampled_discussions = [(df.iloc[idx]["DISCUSSION-ID"], df.iloc[idx]["COMMENTS"]) for idx in selected_indices]

    samples = []
    for discussion_id, comment_list in sampled_discussions:
        for comment in comment_list:
            samples.append({
                'turn': comment['TEXT-CLEAN'],
                'full_discussion': ' '.join([c['TEXT-CLEAN'] for c in comment_list]),
                'discussion_type': 'wiki_interact',
                'discussion_id': discussion_id
            })

    return samples


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


def flatten(xss):
    return [x for xs in xss for x in xs]


def RFC_data_sample():
    df = pd.read_json('../../data/wikidetox-wikiconv/rfc_predecessor_pairs_plain_text (4).json')

    # Get random sample of 5 discussion pairs
    selected_indices = random.sample(range(len(df)), 5)
    sampled_pairs = [(idx, df.iloc[idx]["rfc_predecessor_pairs"]) for idx in selected_indices]

    samples = []
    for original_idx, pair in sampled_pairs:
        rfc_discussion = pair['rfc_discussion_full_text']
        pre_rfc_discussion = pair['predecessor_full_text_until_rfc_started']

        # Use the original IDs from the dataset
        rfc_id = pair['rfc_id']
        predecessor_id = pair['predecessor_id']

        # Get all turns from RFC discussion
        rfc_turns = split_discussion(rfc_discussion)
        for turn in rfc_turns:
            samples.append({
                'turn': turn,
                'full_discussion': rfc_discussion,
                'discussion_type': 'rfc',
                'discussion_id': rfc_id
            })

        # Get all turns from pre-RFC discussion
        pre_rfc_turns = split_discussion(pre_rfc_discussion)
        for turn in pre_rfc_turns:
            samples.append({
                'turn': turn,
                'full_discussion': pre_rfc_discussion,
                'discussion_type': 'pre_rfc',
                'discussion_id': predecessor_id
            })

    return samples


def main():
    # Get samples from both datasets
    interact_samples = wiki_interact_sample()
    rfc_samples = RFC_data_sample()

    # Combine all samples
    all_samples = interact_samples + rfc_samples

    # Convert to DataFrame
    df = pd.DataFrame(all_samples)

    # Print statistics
    print("Summary statistics:")
    print("\nTotal number of turns:", len(df))
    print("\nTurns per discussion type:")
    print(df.groupby('discussion_type').size())
    print("\nNumber of discussions per type:")
    print(df.groupby(['discussion_type', 'discussion_id']).size().groupby(level=0).size())
    print("\nAverage turns per discussion by type:")
    print(df.groupby(['discussion_type', 'discussion_id']).size().groupby(level=0).mean())

    # Save to CSV
    df.to_csv('random_samples.csv', index=False, encoding='utf-8')

    # Save summary statistics to a separate file
    with open('sample_statistics.txt', 'w') as f:
        f.write("Summary Statistics\n")
        f.write("=================\n")
        f.write(f"Total number of turns: {len(df)}\n\n")
        f.write("Turns per discussion type:\n")
        f.write(str(df.groupby('discussion_type').size()) + "\n\n")
        f.write("Turns per individual discussion:\n")
        f.write(str(df.groupby(['discussion_type', 'discussion_id']).size()))


if __name__ == "__main__":
    main()