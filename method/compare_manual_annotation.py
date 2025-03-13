# Author: Thijs Brekhof
# Usage: Comparing annotations by A1 and A2 so A3 can make a judgement about
# what annotations to keep in case of disagreement

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score


def load_annotation_data(annotator1_file, annotator2_file):
    """
    Load and validate the annotation files from both annotators
    """
    ann1_df = pd.read_csv(annotator1_file)
    ann2_df = pd.read_csv(annotator2_file)
    return ann1_df, ann2_df


def find_disagreements(ann1_df, ann2_df, questions):
    """
    Identify where annotators disagree and on which questions
    Returns a list of disagreements with turn numbers and questions
    """
    disagreements = []

    for turn_idx in range(len(ann1_df)):
        turn_disagreements = []
        for question in questions:
            if ann1_df.iloc[turn_idx][question] != ann2_df.iloc[turn_idx][question]:
                turn_disagreements.append(question)

        if turn_disagreements:
            disagreements.append({
                'turn': turn_idx + 1,
                'questions': turn_disagreements,
                'annotator1_answers': [ann1_df.iloc[turn_idx][q] for q in turn_disagreements],
                'annotator2_answers': [ann2_df.iloc[turn_idx][q] for q in turn_disagreements]
            })

    return disagreements


def calculate_overall_class_specific_metrics(ann1_df, ann2_df, questions):
    """Calculate class-specific metrics aggregated across all questions"""

    # Initialize counters for the confusion matrix
    total_yes_yes = 0
    total_yes_no = 0
    total_no_yes = 0
    total_no_no = 0

    # Aggregate counts across all questions
    for question in questions:
        # Add to confusion matrix totals
        total_yes_yes += sum((ann1_df[question] == 'yes') & (ann2_df[question] == 'yes'))
        total_yes_no += sum((ann1_df[question] == 'yes') & (ann2_df[question] == 'no'))
        total_no_yes += sum((ann1_df[question] == 'no') & (ann2_df[question] == 'yes'))
        total_no_no += sum((ann1_df[question] == 'no') & (ann2_df[question] == 'no'))

    # Calculate overall class-specific metrics
    total_agreements = total_yes_yes + total_no_no
    total_disagreements = total_yes_no + total_no_yes
    total_cases = total_agreements + total_disagreements

    # Overall agreement rate
    overall_agreement_rate = total_agreements / total_cases if total_cases > 0 else 0

    # Class-specific agreement rates
    yes_agreement_rate = total_yes_yes / (total_yes_yes + total_yes_no + total_no_yes) if \
        (total_yes_yes + total_yes_no + total_no_yes) > 0 else 0
    no_agreement_rate = total_no_no / (total_no_no + total_yes_no + total_no_yes) if \
        (total_no_no + total_yes_no + total_no_yes) > 0 else 0

    # Return comprehensive metrics
    return {
        'overall_agreement_rate': overall_agreement_rate,
        'yes_agreement_rate': yes_agreement_rate,
        'no_agreement_rate': no_agreement_rate,
        'total_cases': total_cases
    }


def calculate_turn_based_metrics_with_all_no(ann1_df, ann2_df, questions):
    """Calculate turn-based metrics with additional tracking for 'all no' turns"""

    perfect_agreement_counts = 0
    all_no_perfect_agreement_counts = 0
    at_least_one_yes_perfect_agreement_counts = 0
    per_turn_agreement_rates = []

    for i in range(len(ann1_df)):
        # Count agreements for this turn
        agreements_in_turn = sum(ann1_df.iloc[i][q] == ann2_df.iloc[i][q] for q in questions)

        # Calculate agreement rate for this turn
        turn_agreement_rate = agreements_in_turn / len(questions)
        per_turn_agreement_rates.append(turn_agreement_rate)

        # Check if this turn has perfect agreement (all questions agreed upon)
        if agreements_in_turn == len(questions):
            perfect_agreement_counts += 1

            # Check if this was an "all no" agreement
            all_no_in_turn = all(
                (ann1_df.iloc[i][q] == 'no' and ann2_df.iloc[i][q] == 'no')
                for q in questions
            )

            if all_no_in_turn:
                all_no_perfect_agreement_counts += 1
            else:
                at_least_one_yes_perfect_agreement_counts += 1

    # Calculate perfect agreement rate
    perfect_agreement_rate = perfect_agreement_counts / len(ann1_df)

    # Calculate average agreement per turn
    average_agreement_per_turn = sum(per_turn_agreement_rates) / len(per_turn_agreement_rates)

    # Store turn-based metrics
    metrics = {
        'perfect_agreement_rate': perfect_agreement_rate,
        'average_agreement_per_turn': average_agreement_per_turn,
        'turns_with_perfect_agreement': perfect_agreement_counts,
        'turns_with_all_no_perfect_agreement': all_no_perfect_agreement_counts,
        'turns_with_at_least_one_yes_perfect_agreement': at_least_one_yes_perfect_agreement_counts,
        'total_turns': len(ann1_df)
    }

    return metrics


def calculate_agreement_metrics(ann1_df, ann2_df, questions):
    """
    Calculate various inter-annotator agreement metrics with handling for edge cases
    """
    metrics = {
        'overall': {},
        'per_question': {},
    }

    # Calculate overall agreement
    total_decisions = len(ann1_df) * len(questions)
    agreements = sum(
        (ann1_df[q] == ann2_df[q]).sum()
        for q in questions
    )
    metrics['overall']['simple_agreement'] = agreements / total_decisions

    # Calculate overall Cohen's Kappa
    all_ann1 = []
    all_ann2 = []
    for q in questions:
        all_ann1.extend(ann1_df[q])
        all_ann2.extend(ann2_df[q])

    try:
        metrics['overall']['cohens_kappa'] = cohen_kappa_score(all_ann1, all_ann2)
    except:
        metrics['overall']['cohens_kappa'] = 1.0 if metrics['overall']['simple_agreement'] == 1.0 else 0.0

    # Calculate per-question metrics
    for question in questions:
        agreement = (ann1_df[question] == ann2_df[question]).mean()

        # Check if we have perfect agreement or all same values
        ann1_unique = ann1_df[question].nunique()
        ann2_unique = ann2_df[question].nunique()

        if agreement == 1.0:
            kappa = 1.0
        elif ann1_unique == 1 and ann2_unique == 1:
            kappa = 1.0 if ann1_df[question].iloc[0] == ann2_df[question].iloc[0] else 0.0
        elif ann1_unique == 1 or ann2_unique == 1:
            kappa = 0.0
        else:
            try:
                kappa = cohen_kappa_score(ann1_df[question], ann2_df[question])
            except:
                kappa = 0.0

        metrics['per_question'][question] = {
            'simple_agreement': agreement,
            'cohens_kappa': kappa,
            'annotator1_distribution': ann1_df[question].value_counts().to_dict(),
            'annotator2_distribution': ann2_df[question].value_counts().to_dict()
        }

    # Calculate turn-based metrics with all-no tracking
    metrics['turn_based'] = calculate_turn_based_metrics_with_all_no(ann1_df, ann2_df, questions)

    return metrics


def print_results(disagreements, metrics, class_metrics, questions):
    """
    Print formatted results of the analysis with additional distribution information
    """
    print("\n=== INTER-ANNOTATOR AGREEMENT ANALYSIS ===\n")

    # Print overall metrics
    print("Overall Metrics:")
    print(f"Simple Agreement: {metrics['overall']['simple_agreement']:.3f}")
    print(f"Cohen's Kappa: {metrics['overall']['cohens_kappa']:.3f}")

    # Print turn-based metrics with all-no breakdown
    print("\nTurn-Based Metrics:")
    print(f"Perfect Agreement Rate: {metrics['turn_based']['perfect_agreement_rate']:.3f} "
          f"({metrics['turn_based']['turns_with_perfect_agreement']} out of {metrics['turn_based']['total_turns']} turns)")
    print(f"  - All 'no' perfect agreement: {metrics['turn_based']['turns_with_all_no_perfect_agreement']} turns "
          f"({metrics['turn_based']['turns_with_all_no_perfect_agreement'] / metrics['turn_based']['turns_with_perfect_agreement']:.1%} of perfect agreements)")
    print(
        f"  - At least one 'yes' perfect agreement: {metrics['turn_based']['turns_with_at_least_one_yes_perfect_agreement']} turns "
        f"({metrics['turn_based']['turns_with_at_least_one_yes_perfect_agreement'] / metrics['turn_based']['turns_with_perfect_agreement']:.1%} of perfect agreements)")
    print(f"Average Agreement Per Turn: {metrics['turn_based']['average_agreement_per_turn']:.3f}")

    # Print overall class-specific metrics
    print("\n=== OVERALL CLASS-SPECIFIC METRICS ===\n")
    print(f"Total cases: {class_metrics['total_cases']}")

    # Print class-specific agreement rates
    print("\nClass-specific Agreement Rates:")
    print(f"Agreement on 'yes': {class_metrics['yes_agreement_rate']:.3f}")
    print(f"Agreement on 'no': {class_metrics['no_agreement_rate']:.3f}")

    # Print per-question metrics
    print("\nPer-Question Metrics:")
    for question in questions:
        q_metrics = metrics['per_question'][question]
        print(f"\n{question}:")
        print(f"  Simple Agreement: {q_metrics['simple_agreement']:.3f}")
        print(f"  Cohen's Kappa: {q_metrics['cohens_kappa']:.3f}")
        print("  Annotator 1 distribution:", q_metrics['annotator1_distribution'])
        print("  Annotator 2 distribution:", q_metrics['annotator2_distribution'])

    # Print disagreements
    print("\nDetailed Disagreements:")
    print(f"Total number of turns with disagreements: {len(disagreements)}")

    for d in disagreements:
        print(f"\nTurn {d['turn']}:")
        for q, a1, a2 in zip(d['questions'], d['annotator1_answers'], d['annotator2_answers']):
            print(f"  Question: {q}")
            print(f"    Annotator 1: {a1}")
            print(f"    Annotator 2: {a2}")


def main():
    # File paths
    annotator1_file = "results/manual_annotation/Manual Annotation - A1 annotations.csv"
    annotator2_file = "results/manual_annotation/Manual Annotation - A2 annotations.csv"

    # Load data
    ann1_df, ann2_df = load_annotation_data(annotator1_file, annotator2_file)

    # Get question columns (excluding non-question columns)
    exclude_columns = ['Turn', 'Dataset', 'Discussion text', 'Discussion ID']
    questions = [col for col in ann1_df.columns if col not in exclude_columns]

    # Find disagreements
    disagreements = find_disagreements(ann1_df, ann2_df, questions)

    # Calculate agreement metrics
    metrics = calculate_agreement_metrics(ann1_df, ann2_df, questions)

    # Calculate overall class-specific metrics
    class_metrics = calculate_overall_class_specific_metrics(ann1_df, ann2_df, questions)

    # Print results
    print_results(disagreements, metrics, class_metrics, questions)


if __name__ == "__main__":
    main()
