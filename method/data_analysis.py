# Author: Thijs Brekhof
# Usage: analyzing our data and gathering statistics

from collections import Counter
import statistics
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from test_questions import split_discussion


def wiki_interact_stats():
    print("Wiki-interact stats:")
    print("Might take a few seconds to load because of dataset size..\n")
    try:
        # Read the JSON file
        df = pd.read_json(r'C:\Users\Thijs\Documents\Master\Thesis\method\results\sampling-method\sampled_interact_data.jsonl', lines=True)

        # Initialize statistics
        token_counts = []
        levels = []
        users = []
        unique_pages = df['PAGE-TITLE'].unique()

        # Process comments
        for comment_l in df["COMMENTS"]:
            for comment in comment_l:
                levels.append(comment['LEVEL'])
                users.append(comment['USER'])
                token_counts.append(len(comment['TEXT-CLEAN'].split()))

        # Calculate basic statistics
        c = Counter(users)
        d = Counter(levels)

        # Print basic statistics
        print("Dataset Overview:")
        print(f"Unique pages: {len(unique_pages)}")
        print(f"Total amount of discussions: {len(df)}")
        print(f"Comment hierarchy distribution: {dict(d)}")
        print(f"Total amount of conversational turns: {len(token_counts)}")
        print(f"Number of unique users: {len(c)}")
        print(f"Average conversational turns per user: {statistics.mean(c.values()):.2f}")

        print("\nToken Statistics:")
        print(f"Lowest token count: {min(token_counts)}")
        print(f"Highest token count: {max(token_counts)}")
        print(f"Average token count: {statistics.mean(token_counts):.2f}")

        # Create visualization for token distribution
        if token_counts:
            print("\nShowing token distribution visualization...")
            fig, _ = determine_tokenlimit(token_counts, "Token Distribution in Wiki-Fact-Check-Interaction Dataset")
            plt.show()
            plt.close(fig)
            plt.close('all')
    finally:
        plt.close('all')


def rfc_data_stats():
    """
    Calculate statistics for RFC dataset including split discussions.
    """
    try:
        unflattened_df = pd.read_json(
            r'C:\Users\Thijs\Documents\Master\Thesis\data\wikidetox-wikiconv\rfc_predecessor_pairs_plain_text (4).json')

        discussion_rfc_df = pd.json_normalize(unflattened_df['rfc_predecessor_pairs'])

        # Initialize statistics dictionaries
        rfc_stats = {
            'conv_turns': 0,
            'token_counts': [],
            'discussions': set()
        }

        pre_rfc_stats = {
            'conv_turns': 0,
            'token_counts': [],
            'discussions': set()
        }

        # Process each discussion
        for idx, row in discussion_rfc_df.iterrows():
            # Process predecessor discussion
            if pd.notna(row['predecessor_full_text_until_rfc_started']):
                pred_turns = split_discussion(row['predecessor_full_text_until_rfc_started'])
                for turn in pred_turns:
                    pre_rfc_stats['conv_turns'] += 1
                    pre_rfc_stats['token_counts'].append(len(turn.split()))
                pre_rfc_stats['discussions'].add(row['predecessor_id'])

            # Process RFC discussion
            if pd.notna(row['rfc_discussion_full_text']):
                rfc_turns = split_discussion(row['rfc_discussion_full_text'])
                for turn in rfc_turns:
                    rfc_stats['conv_turns'] += 1
                    rfc_stats['token_counts'].append(len(turn.split()))
                rfc_stats['discussions'].add(row['rfc_id'])

        # Calculate and print statistics
        print("RFC data stats:\n")
        print(f"Total RFCs: {len(rfc_stats['discussions'])}")
        print(f"Total predecessor discussions: {len(pre_rfc_stats['discussions'])}")

        print("\nRFC discussion statistics:")
        print(f"Total conversational turns: {rfc_stats['conv_turns']}")

        if rfc_stats['token_counts']:
            print(f"Tokens per turn:")
            print(f"- Average: {statistics.mean(rfc_stats['token_counts']):.2f}")
            print(f"- Minimum: {min(rfc_stats['token_counts'])}")
            print(f"- Maximum: {max(rfc_stats['token_counts'])}")
            print(f"- Median: {statistics.median(rfc_stats['token_counts']):.2f}")

        print("\nPre-RFC discussion statistics:")

        print(f"Total conversational turns: {pre_rfc_stats['conv_turns']}")
        if pre_rfc_stats['token_counts']:
            print(f"Tokens per turn:")
            print(f"- Average: {statistics.mean(pre_rfc_stats['token_counts']):.2f}")
            print(f"- Minimum: {min(pre_rfc_stats['token_counts'])}")
            print(f"- Maximum: {max(pre_rfc_stats['token_counts'])}")
            print(f"- Median: {statistics.median(pre_rfc_stats['token_counts']):.2f}")

        figures = []
        if rfc_stats['token_counts']:
            print("\nRFC Discussion Token Distribution:")
            fig_rfc, _ = determine_tokenlimit(
                rfc_stats['token_counts'],
                "Token Distribution in RFC Discussions"
            )
            figures.append(fig_rfc)

        if pre_rfc_stats['token_counts']:
            print("\nPre-RFC Discussion Token Distribution:")
            fig_pre_rfc, _ = determine_tokenlimit(
                pre_rfc_stats['token_counts'],
                "Token Distribution in Pre-RFC Discussions"
            )
            figures.append(fig_pre_rfc)

        # Show all figures at once
        plt.show()


    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def determine_tokenlimit(token_lengths, title=None):
    """
    Create distribution visualizations for token lengths.
    Args:
        token_lengths: List of token counts
        title: Optional title for the entire figure
    Returns the figure and axes for external plotting control.
    """
    # Calculate statistics
    mean = np.mean(token_lengths)
    median = np.median(token_lengths)
    percentiles = np.percentile(token_lengths, [90, 95, 99])

    # Calculate CDF
    sorted_lengths = np.sort(token_lengths)
    cumulative = np.cumsum(sorted_lengths) / np.sum(sorted_lengths)

    # Create figure and adjust subplot parameters to make room for title
    fig = plt.figure(figsize=(12, 9))
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)

        plt.subplots_adjust(top=0.9)

    # Create subplots
    gs = fig.add_gridspec(2, 2)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])

    # Plot histogram
    axs[0, 0].hist(token_lengths, bins=50)
    axs[0, 0].set_xlim(0, 1000)
    axs[0, 0].set_title('Histogram of Token Lengths')

    # Plot boxplot
    axs[0, 1].boxplot(token_lengths)
    axs[0, 1].set_title('Box Plot of Token Lengths')

    # Plot CDF
    axs[1, 0].plot(sorted_lengths, cumulative)
    axs[1, 0].set_xlim(0, 1000)
    axs[1, 0].set_title('Cumulative Distribution Function')

    # Plot samples covered
    axs[1, 1].plot(sorted_lengths, range(len(sorted_lengths)))
    axs[1, 1].set_xlim(0, 1000)
    axs[1, 1].set_title('Samples Covered vs Token Limit')

    plt.tight_layout()

    # Print statistics
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"90th, 95th, 99th percentiles: {percentiles}")

    return fig, axs


def main():
    wiki_interact_stats()

    rfc_data_stats()


main()
