# Author: Thijs Brekhof
# Usage: Comparing the classification results of the models to our manual annotation

import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 300


def load_data(manual_file, model_files):
    """
    Load and validate the manual and model classification files
    """
    manual_df = pd.read_csv(manual_file)
    model_dfs = []
    for f in model_files:
        model_dfs.append(pd.read_csv(f))
    return manual_df, model_dfs


def analyze_results(manual_df, model_df, questions):
    """Calculate overall metrics treating each turn as multi-label classification"""
    total_turns = len(manual_df)
    turn_level_precision = []
    turn_level_recall = []
    turn_level_f1 = []

    # For micro metrics (aggregating across all turns and questions)
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0

    # For macro metrics (per question)
    question_precision = []
    question_recall = []
    question_f1 = []

    # Calculate individual turn metrics (existing code)
    for idx in range(total_turns):
        # Get true and predicted labels for a turn
        true_labels = set()
        pred_labels = set()

        for q_idx, question in enumerate(questions):
            if manual_df.iloc[idx][question] == 'yes':
                true_labels.add(q_idx)
            if model_df.iloc[idx][question] == 'yes':
                pred_labels.add(q_idx)

        # Calculate turn-level metrics
        if len(true_labels) == 0 and len(pred_labels) == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        elif len(pred_labels) == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            intersection = len(true_labels.intersection(pred_labels))
            precision = intersection / len(pred_labels) if pred_labels else 0
            recall = intersection / len(true_labels) if true_labels else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        turn_level_precision.append(precision)
        turn_level_recall.append(recall)
        turn_level_f1.append(f1)

    # Calculate micro-average metrics (aggregating all predictions)
    for q in questions:
        for idx in range(total_turns):
            true_value = manual_df.iloc[idx][q]
            pred_value = model_df.iloc[idx][q]

            if true_value == 'yes' and pred_value == 'yes':
                all_true_positives += 1
            elif true_value == 'no' and pred_value == 'yes':
                all_false_positives += 1
            elif true_value == 'yes' and pred_value == 'no':
                all_false_negatives += 1

    micro_precision = all_true_positives / (all_true_positives + all_false_positives) if (
                                                                                                     all_true_positives + all_false_positives) > 0 else 0
    micro_recall = all_true_positives / (all_true_positives + all_false_negatives) if (
                                                                                                  all_true_positives + all_false_negatives) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                    micro_precision + micro_recall) > 0 else 0

    # Calculate macro-average metrics (average of per-question metrics)
    for q in questions:
        q_true_positives = 0
        q_false_positives = 0
        q_false_negatives = 0

        for idx in range(total_turns):
            true_value = manual_df.iloc[idx][q]
            pred_value = model_df.iloc[idx][q]

            if true_value == 'yes' and pred_value == 'yes':
                q_true_positives += 1
            elif true_value == 'no' and pred_value == 'yes':
                q_false_positives += 1
            elif true_value == 'yes' and pred_value == 'no':
                q_false_negatives += 1

        q_precision = q_true_positives / (q_true_positives + q_false_positives) if (
                                                                                               q_true_positives + q_false_positives) > 0 else 0
        q_recall = q_true_positives / (q_true_positives + q_false_negatives) if (
                                                                                            q_true_positives + q_false_negatives) > 0 else 0
        q_f1 = 2 * q_precision * q_recall / (q_precision + q_recall) if (q_precision + q_recall) > 0 else 0

        question_precision.append(q_precision)
        question_recall.append(q_recall)
        question_f1.append(q_f1)

    macro_precision = np.mean(question_precision)
    macro_recall = np.mean(question_recall)
    macro_f1 = np.mean(question_f1)

    return {
        'precision': np.mean(turn_level_precision),
        'recall': np.mean(turn_level_recall),
        'f1': np.mean(turn_level_f1),
        'perfect_turns': sum(1 for f1 in turn_level_f1 if f1 == 1.0),
        'failed_turns': sum(1 for f1 in turn_level_f1 if f1 == 0.0),
        'total_turns': total_turns,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


def print_metrics_summary(results, model_name):
    """Print the multi-label metrics summary"""
    formatted_name = format_model_name(model_name)
    print(f"\n=== Results for {formatted_name} ===")

    print("\nSample-averaged Classification Metrics (average across turns):")
    print(f"Average Precision: {results['precision']:.3f}")
    print(f"Average Recall: {results['recall']:.3f}")
    print(f"Average F1: {results['f1']:.3f}")

    print("\nMicro-averaged Classification Metrics (aggregated across all predictions):")
    print(f"Micro Precision: {results['micro_precision']:.6f}")
    print(f"Micro Recall: {results['micro_recall']:.3f}")
    print(f"Micro F1: {results['micro_f1']:.3f}")


    if np.isnan(results['micro_precision']) or np.isinf(results['micro_precision']):
        print("Warning: Numeric error in precision calculation")

    print("\nMacro-averaged Classification Metrics (average across questions):")
    print(f"Macro Precision: {results['macro_precision']:.3f}")
    print(f"Macro Recall: {results['macro_recall']:.3f}")
    print(f"Macro F1: {results['macro_f1']:.3f}")

    print(f"\nTurn-level Statistics:")
    print(f"Total turns: {results['total_turns']}")
    perfect_pct = results['perfect_turns'] / results['total_turns']
    failed_pct = results['failed_turns'] / results['total_turns']
    partial_turns = results['total_turns'] - results['perfect_turns'] - results['failed_turns']
    partial_pct = partial_turns / results['total_turns']

    print(f"Perfect predictions (F1=1.0): {results['perfect_turns']} ({perfect_pct:.1%})")
    print(f"Partial predictions (0<F1<1): {partial_turns} ({partial_pct:.1%})")
    print(f"Failed predictions (F1=0.0): {results['failed_turns']} ({failed_pct:.1%})")


def create_multilabel_metrics_table(manual_df, model_df, questions, fig_num):
    """Create a comprehensive table showing multi-label classification metrics"""
    plt.figure(fig_num, figsize=(15, 20))
    plt.clf()

    # Filter out unused questions
    active_questions = []
    filtered_data = []

    for q in questions:
        true_labels = manual_df[q]
        pred_labels = model_df[q]

        # Check if question appears in either manual annotations or predictions
        has_manual_yes = sum(true_labels == 'yes') > 0
        has_pred_yes = sum(pred_labels == 'yes') > 0

        if has_manual_yes or has_pred_yes:
            active_questions.append(q)

    # Calculate overall multi-label metrics for active questions only
    y_true = manual_df[active_questions].apply(lambda x: x == 'yes').astype(int)
    y_pred = model_df[active_questions].apply(lambda x: x == 'yes').astype(int)

    instance_based = {
        'exact_match': sum(
            (y_true.values == y_pred.values).all(axis=1)
        ) / len(y_true),
        'hamming_loss': (
                            (y_true.values != y_pred.values).sum()
                        ) / (len(y_true) * len(active_questions))
    }

    # For each active question
    for q in active_questions:
        true_labels = manual_df[q]
        pred_labels = model_df[q]

        # Calculate occurrences
        total_actual = sum(true_labels == 'yes')
        total_predicted = sum(pred_labels == 'yes')
        true_positives = sum((true_labels == 'yes') & (pred_labels == 'yes'))

        # Calculate metrics for 'yes' predictions
        yes_precision = true_positives / total_predicted if total_predicted > 0 else 0
        yes_recall = true_positives / total_actual if total_actual > 0 else 0
        yes_f1 = 2 * (yes_precision * yes_recall) / (yes_precision + yes_recall) if (
                                                                                                yes_precision + yes_recall) > 0 else 0

        # Calculate metrics for 'no' predictions
        true_negatives = sum((true_labels == 'no') & (pred_labels == 'no'))
        total_actual_no = sum(true_labels == 'no')
        total_predicted_no = sum(pred_labels == 'no')

        no_precision = true_negatives / total_predicted_no if total_predicted_no > 0 else 0
        no_recall = true_negatives / total_actual_no if total_actual_no > 0 else 0
        no_f1 = 2 * (no_precision * no_recall) / (no_precision + no_recall) if (no_precision + no_recall) > 0 else 0

        filtered_data.append([
            total_actual,  # Total actual occurrences
            f"{true_positives}/{total_predicted}",  # Correct/Total predictions
            f"{yes_precision:.3f}",  # Yes precision
            f"{yes_recall:.3f}",  # Yes recall
            f"{yes_f1:.3f}",  # Yes F1
            f"{no_precision:.3f}",  # No precision
            f"{no_recall:.3f}",  # No recall
            f"{no_f1:.3f}"  # No F1
        ])

    # Create column labels
    columns = [
        'Total\nActual',
        'Correct/Total\nPredictions',
        'Yes\nPrecision',
        'Yes\nRecall',
        'Yes\nF1',
        'No\nPrecision',
        'No\nRecall',
        'No\nF1'
    ]

    # Create row labels for active questions only
    rows = [f'Q{questions.index(q) + 1}' for q in active_questions]

    # Create table
    table = plt.table(
        cellText=filtered_data,
        rowLabels=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]
    )

    # Add overall metrics as title
    plt.suptitle('Multi-label Classification Metrics', fontsize=12, y=0.95)
    plt.title(
        f'Overall Metrics:\n'
        f'Exact Match Ratio: {instance_based["exact_match"]:.3f}\n'
        f'Hamming Loss: {instance_based["hamming_loss"]:.3f}\n'
        f'Active Questions: {len(active_questions)}/{len(questions)}',
        fontsize=10,
        pad=20
    )

    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    # Remove axes
    plt.axis('off')
    plt.tight_layout()


def create_visualizations(model_name, questions, manual_df, model_df, fig_num):
    """Create confusion matrix and error distribution visualizations"""
    formatted_name = format_model_name(model_name)

    # First figure - Confusion Matrix
    plt.figure(fig_num, figsize=(6, 5))
    plt.clf()

    # Combine all predictions for confusion matrix
    all_true = []
    all_pred = []
    for q in questions:
        all_true.extend(manual_df[q])
        all_pred.extend(model_df[q])

    # Create confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=['yes', 'no'])

    plt.title(f'{formatted_name}\nConfusion Matrix', pad=20)
    sns.heatmap(

        cm,
        annot=True,
        fmt='d',
        cmap='rocket_r',
        xticklabels=['yes', 'no'],
        yticklabels=['yes', 'no'],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # Second figure - Error Distribution by Question
    plt.figure(fig_num + 1, figsize=(8, 10))
    plt.clf()

    # Calculate per-question metrics
    question_metrics = []
    for i, q in enumerate(questions, 1):
        true_labels = manual_df[q]
        pred_labels = model_df[q]

        tp = sum((true_labels == 'yes') & (pred_labels == 'yes'))
        fp = sum((true_labels == 'no') & (pred_labels == 'yes'))
        fn = sum((true_labels == 'yes') & (pred_labels == 'no'))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        question_metrics.append({
            'Question_Num': i,
            'FP': fp,
            'FN': fn,
            'F1': f1
        })

    # Convert to DataFrame and sort by question number
    metrics_df = pd.DataFrame(question_metrics)
    metrics_df = metrics_df.sort_values('Question_Num', ascending=False)

    plt.title(f'{formatted_name}\nError Distribution by Question', pad=20)
    x = np.arange(len(questions))
    width = 0.35

    plt.barh(x - width / 2, metrics_df['FP'], width, label='False Positives', color='lightcoral')
    plt.barh(x + width / 2, metrics_df['FN'], width, label='False Negatives', color='lightblue')

    plt.yticks(x, metrics_df['Question_Num'], fontsize=8)
    plt.xlabel('Number of Errors')
    plt.legend()
    plt.tight_layout()

    # Third figure - Error Distribution by Turn
    plt.figure(fig_num + 2, figsize=(8, 10))
    plt.clf()

    # Calculate per-turn metrics
    turn_metrics = []
    for turn_idx in range(len(manual_df)):
        fp = 0
        fn = 0
        for q in questions:
            if manual_df.iloc[turn_idx][q] == 'no' and model_df.iloc[turn_idx][q] == 'yes':
                fp += 1
            if manual_df.iloc[turn_idx][q] == 'yes' and model_df.iloc[turn_idx][q] == 'no':
                fn += 1

        if fp > 0 or fn > 0:
            turn_metrics.append({
                'Turn': turn_idx + 1,
                'FP': fp,
                'FN': fn
            })

    # multi-label metrics table
    create_multilabel_metrics_table(manual_df, model_df, questions, fig_num + 3)

    plt.show()


def print_multilabel_metrics_table(manual_df, model_df, questions, model_name):
    """Print a text-based version of the multilabel classification metrics table"""
    formatted_name = format_model_name(model_name)
    print(f"\n=== Multilabel Metrics Table for {formatted_name} ===\n")

    # Filter out unused questions
    active_questions = []
    for q in questions:
        true_labels = manual_df[q]
        pred_labels = model_df[q]

        # Check if question appears in either manual annotations or predictions
        has_manual_yes = sum(true_labels == 'yes') > 0
        has_pred_yes = sum(pred_labels == 'yes') > 0

        if has_manual_yes or has_pred_yes:
            active_questions.append(q)

    # Calculate overall multi-label metrics for active questions only
    y_true = manual_df[active_questions].apply(lambda x: x == 'yes').astype(int)
    y_pred = model_df[active_questions].apply(lambda x: x == 'yes').astype(int)

    exact_match = sum((y_true.values == y_pred.values).all(axis=1)) / len(y_true)
    hamming_loss = ((y_true.values != y_pred.values).sum()) / (len(y_true) * len(active_questions))

    # Print overall metrics
    print(f"Overall Metrics:")
    print(f"  Exact Match Ratio: {exact_match:.3f}")
    print(f"  Hamming Loss: {hamming_loss:.3f}")
    print(f"  Active Questions: {len(active_questions)}/{len(questions)}")

    # Print header for table
    print("\n{:<5} {:<10} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Q#", "Total", "Correct/Total", "Yes", "Yes", "Yes", "No", "No", "No"))
    print("{:<5} {:<10} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "", "Actual", "Predictions", "Precision", "Recall", "F1", "Precision", "Recall", "F1"))
    print("-" * 95)

    # For each active question
    for q_idx, q in enumerate(active_questions):
        q_num = questions.index(q) + 1
        true_labels = manual_df[q]
        pred_labels = model_df[q]

        # Calculate occurrences
        total_actual = sum(true_labels == 'yes')
        total_predicted = sum(pred_labels == 'yes')
        true_positives = sum((true_labels == 'yes') & (pred_labels == 'yes'))

        # Calculate metrics for 'yes' predictions
        yes_precision = true_positives / total_predicted if total_predicted > 0 else 0
        yes_recall = true_positives / total_actual if total_actual > 0 else 0
        yes_f1 = 2 * (yes_precision * yes_recall) / (yes_precision + yes_recall) if (
                                                                                                yes_precision + yes_recall) > 0 else 0

        # Calculate metrics for 'no' predictions
        true_negatives = sum((true_labels == 'no') & (pred_labels == 'no'))
        total_actual_no = sum(true_labels == 'no')
        total_predicted_no = sum(pred_labels == 'no')

        no_precision = true_negatives / total_predicted_no if total_predicted_no > 0 else 0
        no_recall = true_negatives / total_actual_no if total_actual_no > 0 else 0
        no_f1 = 2 * (no_precision * no_recall) / (no_precision + no_recall) if (no_precision + no_recall) > 0 else 0

        # Print row
        print("{:<5} {:<10} {:<15} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
            f"Q{q_num}",
            total_actual,
            f"{true_positives}/{total_predicted}",
            yes_precision,
            yes_recall,
            yes_f1,
            no_precision,
            no_recall,
            no_f1
        ))

def format_model_name(filename):
    """Convert filename to readable model name and setting"""
    # Remove path and 'results_' prefix
    base = filename.split('/')[-1].replace('results_', '').replace('.csv', '')

    if 'Llama' in base:
        model = 'Llama 3.1-8B-Instruct'
    elif 'Phi-3.5-mini' in base:
        model = 'Phi-3.5-mini-instruct'
    elif 'Phi-3-medium' in base:
        model = 'Phi-3-medium-128k-instruct'
    else:
        model = base

    # Extract setting
    setting = 'Full' if 'full' in base.lower() else 'Turn'

    return f"{model} & {setting}"


def main():
    manual_results = "results/manual_annotation/Manual Annotation - A3 annotations.csv"
    model_results = [
        # "results/manual_annotation/results_Llama-3.1-8B-Instruct_turn_subsample.csv",
        # "results/manual_annotation/results_Llama-3.1-8B-Instruct_full_subsample.csv",
        # "results/manual_annotation/results_Phi-3.5-mini-instruct_turn_subsample.csv",
        # "results/manual_annotation/results_Phi-3.5-mini-instruct_full_subsample.csv",
        # "results/manual_annotation/results_Phi-3-medium-128k-instruct_turn_subsample.csv",
        # "results/manual_annotation/results_Phi-3-medium-128k-instruct_full_subsample.csv",
        # "results/manual_annotation/deepseek/results_deepseek_chat_turn.csv",
        "results/manual_annotation/deepseek/results_deepseek_chat_full.csv"

    ]

    manual_df, model_df_list = load_data(manual_results, model_results)

    questions = [col for col in manual_df.columns
                 if col not in ['Turn', 'Dataset', 'Discussion ID', 'Discussion text']]

    for idx, model_df in enumerate(model_df_list):
        model_name = model_results[idx].split('results_')[-1].split('.csv')[0]
        results = analyze_results(manual_df, model_df, questions)

        # Print the p/r/f1 scores of the models
        # print_metrics_summary(results, model_name)

        # print the multilabel metrics table
        print_multilabel_metrics_table(manual_df, model_df, questions, model_name)

       # create_visualizations(model_name, questions, manual_df, model_df, idx + 1)


if __name__ == "__main__":
    main()
