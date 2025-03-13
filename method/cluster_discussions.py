# Author: Thijs Brekhof
# Usage: script to first summarize discussions, cluster these summarizations and finally obtain a subset of all the
# discussions (1 per cluster) to have a subsample of data that is varied in topic.

import random
import pandas as pd
from sklearn.cluster import KMeans
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from sentence_transformers import SentenceTransformer
import json
import os
from tqdm import tqdm
import pickle


def save_checkpoint(data, filename):
    """Save checkpoint to file"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_checkpoint(filename):
    """Load checkpoint from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


def batch_summarize_with_checkpoints(texts, discussion_ids, tokenizer, model,
                                     batch_size=64, checkpoint_size=5000,
                                     checkpoint_dir='checkpoints'):
    """Summarizes texts in batches with periodic checkpointing"""

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize or load from last checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, 'summaries_checkpoint.pkl')
    checkpoint_data = load_checkpoint(checkpoint_file)

    if checkpoint_data:
        summaries_dict = checkpoint_data['summaries']
        start_idx = checkpoint_data['last_idx'] + 1
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        summaries_dict = {}
        start_idx = 0

    # Process in batches with progress bar
    for i in tqdm(range(start_idx, len(texts), batch_size)):
        batch_end = min(i + batch_size, len(texts))
        batch_texts = texts[i:batch_end]
        batch_ids = discussion_ids[i:batch_end]

        # Generate summaries for batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        summary_ids = model.generate(**inputs, max_length=30, length_penalty=2.0,
                                     num_beams=4, early_stopping=True)
        batch_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

        # Store summaries with their IDs
        for id_, summary in zip(batch_ids, batch_summaries):
            summaries_dict[id_] = summary

        # Save checkpoint every checkpoint_size discussions
        if batch_end // checkpoint_size > (batch_end - batch_size) // checkpoint_size:
            try:
                checkpoint_data = {
                    'summaries': summaries_dict,
                    'last_idx': batch_end - 1
                }
                save_checkpoint(checkpoint_data, checkpoint_file)
                print(f"\nCheckpoint saved at index {batch_end}")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")

    return summaries_dict


def cluster_and_sample_discussions(discussions, embedding_model, tokenizer, model, num_clusters):
    """
    Summarizes discussions, clusters them, and samples one discussion per cluster
    """
    # Unzip discussions into IDs and texts
    discussion_ids, texts = zip(*discussions)

    # Generate summaries with checkpointing
    print("Generating summaries...")
    summaries_dict = batch_summarize_with_checkpoints(
        texts,
        discussion_ids,
        tokenizer,
        model
    )

    # Convert summaries dict to list maintaining order
    summaries = [summaries_dict[id_] for id_ in discussion_ids]

    # Generate embeddings for summaries
    print("Generating embeddings...")
    embeddings = embedding_model.encode(summaries, convert_to_numpy=True)

    # Apply K-means clustering
    print(f"Clustering into {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=min(num_clusters, len(texts)),
                    random_state=42,
                    n_init=3)
    clusters = kmeans.fit_predict(embeddings)

    # Organize discussions by cluster
    clustered_discussions = {i: [] for i in range(max(clusters) + 1)}
    for idx, cluster in enumerate(clusters):
        clustered_discussions[cluster].append({
            'discussion_id': discussion_ids[idx],
            'text': texts[idx],
            'summary': summaries[idx]
        })

    # Sample one discussion from each cluster
    sampled_discussions = []
    for cluster in clustered_discussions:
        if clustered_discussions[cluster]:
            sampled_discussions.append(random.choice(clustered_discussions[cluster]))

    return sampled_discussions, embeddings, clusters


def select_representative_discussions(clustered_summaries, num_samples=5):
    """Selects a diverse subset of discussions from each cluster"""
    selected_discussions = {}
    for cluster, discussions in clustered_summaries.items():
        if len(discussions) <= num_samples:
            selected_discussions[cluster] = discussions
        else:
            selected_discussions[cluster] = random.sample(discussions, num_samples)
    return selected_discussions


def get_fact_check_interaction_data(file_path):
    """
    Process Wiki-Fact-check-Interaction data with full discussion context.
    Returns list of tuples (discussion_id, combined_text)
    """
    discussions = []

    # Read the JSONL file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Parse each line as a separate JSON object
                data = json.loads(line)

                # Extract discussion ID and page title
                discussion_id = data['DISCUSSION-ID']
                page_title = data['PAGE-TITLE']

                # Get the first comment (comment with lowest COMMENT-NR)
                comments = data['COMMENTS']
                first_comment = min(comments, key=lambda x: x['COMMENT-NR'])
                first_comment_text = first_comment['TEXT-CLEAN']

                # Combine title and first comment for context
                combined_text = f"Title: {page_title}\nFirst Comment: {first_comment_text}"

                # Add to discussions list
                discussions.append((discussion_id, combined_text))

            except Exception as e:
                print(f"Unexpected error processing line: {e}")
                continue

    print(f"Successfully loaded {len(discussions)} discussions")
    return discussions


def main():
    # Load and process data
    print("Loading data...")
    discussions = get_fact_check_interaction_data('Wiki-Fact-check-Interaction.jsonl')

    # Load models
    print("Loading models...")
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Cluster and sample discussions - Total number of clusters = total discussions to keep. This number is derived
    # from 5x the turns in RFC discussion (7761) = 38.805 divided by the average length of turns in interact data
    # (5.54) = 7004
    num_clusters = 7004
    sampled_discussions, embeddings, clusters = cluster_and_sample_discussions(
        discussions, embedding_model, tokenizer, model, num_clusters
    )

    # Save results
    results_df = pd.DataFrame(sampled_discussions)
    results_df.to_csv('sampled_discussions.csv', index=False)
    print(f"Sampled {len(sampled_discussions)} discussions")


if __name__ == "__main__":
    main()
