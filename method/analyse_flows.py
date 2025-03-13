# Author: Thijs Brekhof
# Usage: Analyses discussion data by extracting what questions categories often follow each other
# in successful vs unsuccessful discussions (RFC)

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby


# Function to remove consecutive duplicates
def remove_consecutive_duplicates(sequence):
    if not sequence:
        return []
    return [k for k, g in groupby(sequence)]


def extract_flows(df, question_to_category):
    # Group by discussion and source
    discussions = {}

    for index, source in zip(df['original_index'], df['source_column']):
        if (index, source) not in discussions:
            discussions[(index, source)] = []

    # For each discussion, extract the flow of question categories
    flows = {}

    for (index, source), turns in discussions.items():
        flow_key = f"{index}_{source}"
        flows[flow_key] = {
            'original_index': index,
            'source': source,
            'type': 'RFC' if 'rfc_discussion' in source else 'Predecessor',
            'original_flow': [],
            'change_flow': [],
            'noloops_flow': [],
            'turns_with_questions': 0,
            'total_turns': 0
        }

    # Iterate through each row/turn
    for _, row in df.iterrows():
        index = row['original_index']
        source = row['source_column']
        flow_key = f"{index}_{source}"

        # Increment total turns
        flows[flow_key]['total_turns'] += 1

        # Get the categories addressed in this turn
        addressed_categories = set()

        # Check each question column
        question_columns = [col for col in df.columns if col in question_to_category]
        for question in question_columns:
            if row[question].lower() == 'yes':
                category = question_to_category[question]
                addressed_categories.add(category)

        # If any categories were addressed, add them to the flow
        if addressed_categories:
            flows[flow_key]['turns_with_questions'] += 1
            # Sort categories to ensure consistent order when multiple categories appear in one turn
            sorted_categories = sorted(list(addressed_categories))
            flows[flow_key]['original_flow'].extend(sorted_categories)

    # Apply flow transformations
    for flow_key in flows:
        flows[flow_key]['change_flow'] = remove_consecutive_duplicates(flows[flow_key]['original_flow'])

        # Apply NoLoops transformation
        noloops = []
        i = 0
        while i < len(flows[flow_key]['change_flow']):
            # Check if this sequence appears later
            found_loop = False
            for j in range(i + 2, len(flows[flow_key]['change_flow'])):
                if j + (j - i) <= len(flows[flow_key]['change_flow']):
                    if flows[flow_key]['change_flow'][i:j] == flows[flow_key]['change_flow'][j:j + (j - i)]:
                        # Found a loop, skip to after it
                        noloops.extend(flows[flow_key]['change_flow'][i:j])
                        i = j + (j - i)
                        found_loop = True
                        break
            if not found_loop:
                noloops.append(flows[flow_key]['change_flow'][i])
                i += 1

        flows[flow_key]['noloops_flow'] = noloops

    return flows


def get_question_to_category_mapping():
    category_mapping = {
        'Source Credibility': [
            "What is the source's established reputation and standing within its field of expertise?",
            "Does the author have appropriate expertise or qualifications?",
            "Has the source undergone peer review or editorial oversight?",
            "Is the source free from bias or conflicts of interest?",
            "Is the publisher of the source reputable?",
            "Are there any known retractions orcorrections associated with this source?",
            "Is there transparency about funding sources?",
            "Has the source’s reliability been consistent throughout its history?",
            "Are all made claims supported by a source?"],

        'Verifiability': [
            "Can the information in the source be independently verified?"
            "Is the source based on empirical evidence or firsthand data?",
            "Is the source properly cited and accessible to readers?",
            "Is the information presented by the source up-to-date?",
            "Do other reputable sources (regularly) reference this source?",
            "Are there concrete instances where this source has been proven accurate?",
            "How well are the source's methods and findings preserved?",
            "Does the source properly distinguish between correlation and causation?"],

        'Neutrality and Objectivity': [
            "Is the source known for its objectivity or neutrality?",
            "Does the source reflect the consensus of experts on the subject?",
            "Does the source avoid advocacy or promoting a specific agenda?",
            "Does the source acknowledge and address changes in their positions over time?",
            "Does the source present multiple viewpoints or a balanced perspective?",
            "How does the source handle contradictory evidence?"],

        'Relevance and Applicability': [
            "Is this source directly relevant to the claim being made?",
            "Is the source appropriate for the context of the article?",
            "Does the source provide the correct level of detail or depth for this claim?",
            "Is the source's level of expertise and depth appropriate for how we're using it in this specific context",
            "Does the source base its claims on sufficient examples / cases?",
            "Is the source's coverage representative of the full topic?"],

        'Original Research and Synthesis': [
            "Is this source presenting new or original research?",
            "Does this source draw its own conclusions, or is it summarizing existing knowledge?",
            "Does the claim in the article match what the source explicitly states, or is there interpretation "
            "involved?",
            "Are multiple sources combined in a way that introduces original analysis?"],

        'Policy Compliance and Formatting': [
            "Does the source meet Wikipedia’s standards for reliable sourcing?",
            "Is the citation correctly formatted and detailed?",
            "Does the source comply with Wikipedia’s policy on primary, secondary, and tertiary sources?"],

        'Disagreement and Conflict Resolution': [
            "What do other editors say about the reliability of this source?",
            "What do experts say about the reliability of this source?",
            "Can we reach a consensus on whether this source meets Wikipedia’s standards?",
            "Are there alternative sources we could use that are more reliable?",
            "Are there any previous consensus discussions about this source?",
            "Are there exceptional circumstances affecting the source's reliability?"]
    }

    # Create a reverse mapping from question to category
    question_to_category = {}
    for category, questions_list in category_mapping.items():
        for question in questions_list:
            question_to_category[question] = category

    return question_to_category


def analyze_flows(flows):
    # Separate predecessor and RFC flows
    predecessor_flows = {k: v for k, v in flows.items() if v['type'] == 'Predecessor'}
    rfc_flows = {k: v for k, v in flows.items() if v['type'] == 'RFC'}

    # Analysis 1: Flow length statistics
    length_stats = {
        'Predecessor': {
            'original': np.mean([len(v['original_flow']) for v in predecessor_flows.values()]),
            'change': np.mean([len(v['change_flow']) for v in predecessor_flows.values()]),
            'noloops': np.mean([len(v['noloops_flow']) for v in predecessor_flows.values()])
        },
        'RFC': {
            'original': np.mean([len(v['original_flow']) for v in rfc_flows.values()]),
            'change': np.mean([len(v['change_flow']) for v in rfc_flows.values()]),
            'noloops': np.mean([len(v['noloops_flow']) for v in rfc_flows.values()])
        }
    }

    # Analysis 2: Most common flows (using change flow)
    common_flows = {
        'Predecessor': Counter([tuple(v['change_flow']) for v in predecessor_flows.values() if v['change_flow']]),
        'RFC': Counter([tuple(v['change_flow']) for v in rfc_flows.values() if v['change_flow']])
    }

    # Analysis 3: Category prevalence
    category_counts = {
        'Predecessor': Counter([cat for v in predecessor_flows.values() for cat in v['original_flow']]),
        'RFC': Counter([cat for v in rfc_flows.values() for cat in v['original_flow']])
    }

    # Analysis 4: Transitions between categories
    transitions = {
        'Predecessor': Counter(),
        'RFC': Counter()
    }

    for flow_type in ['Predecessor', 'RFC']:
        flows_of_type = predecessor_flows if flow_type == 'Predecessor' else rfc_flows
        for v in flows_of_type.values():
            for i in range(len(v['change_flow']) - 1):
                transition = (v['change_flow'][i], v['change_flow'][i + 1])
                transitions[flow_type][transition] += 1

    # Analysis 5: Percentage of turns with questions
    question_coverage = {
        'Predecessor': sum(v['turns_with_questions'] for v in predecessor_flows.values()) /
                       sum(v['total_turns'] for v in predecessor_flows.values()),
        'RFC': sum(v['turns_with_questions'] for v in rfc_flows.values()) /
               sum(v['total_turns'] for v in rfc_flows.values())
    }

    return {
        'length_stats': length_stats,
        'common_flows': common_flows,
        'category_counts': category_counts,
        'transitions': transitions,
        'question_coverage': question_coverage
    }


def format_results(analysis_results):
    # Table 1: Flow Length Statistics
    length_df = pd.DataFrame(analysis_results['length_stats']).T
    length_df.columns = ['Original Flow', 'Change Flow', 'NoLoops Flow']
    length_df['Discussion Type'] = length_df.index
    length_df = length_df[['Discussion Type', 'Original Flow', 'Change Flow', 'NoLoops Flow']]

    # Table 2: Most Common Flows (top 10)
    pred_common = analysis_results['common_flows']['Predecessor'].most_common(10)
    rfc_common = analysis_results['common_flows']['RFC'].most_common(10)

    common_flows_data = []
    for i in range(max(len(pred_common), len(rfc_common))):
        row = {}
        if i < len(pred_common):
            row['Predecessor Flow'] = ' → '.join(pred_common[i][0])
            row['Predecessor Count'] = pred_common[i][1]
            row['Predecessor %'] = pred_common[i][1] / sum(
                analysis_results['common_flows']['Predecessor'].values()) * 100
        else:
            row['Predecessor Flow'] = ''
            row['Predecessor Count'] = ''
            row['Predecessor %'] = ''

        if i < len(rfc_common):
            row['RFC Flow'] = ' → '.join(rfc_common[i][0])
            row['RFC Count'] = rfc_common[i][1]
            row['RFC %'] = rfc_common[i][1] / sum(analysis_results['common_flows']['RFC'].values()) * 100
        else:
            row['RFC Flow'] = ''
            row['RFC Count'] = ''
            row['RFC %'] = ''

        common_flows_data.append(row)

    common_flows_df = pd.DataFrame(common_flows_data)

    # Table 3: Category Prevalence
    category_data = []
    all_categories = set(analysis_results['category_counts']['Predecessor'].keys()) | set(
        analysis_results['category_counts']['RFC'].keys())

    for category in sorted(all_categories):
        pred_count = analysis_results['category_counts']['Predecessor'].get(category, 0)
        rfc_count = analysis_results['category_counts']['RFC'].get(category, 0)

        total_pred = sum(analysis_results['category_counts']['Predecessor'].values())
        total_rfc = sum(analysis_results['category_counts']['RFC'].values())

        category_data.append({
            'Category': category,
            'Predecessor Count': pred_count,
            'Predecessor %': pred_count / total_pred * 100 if total_pred > 0 else 0,
            'RFC Count': rfc_count,
            'RFC %': rfc_count / total_rfc * 100 if total_rfc > 0 else 0,
            'Difference (RFC - Pred) %': (rfc_count / total_rfc * 100 if total_rfc > 0 else 0) -
                                         (pred_count / total_pred * 100 if total_pred > 0 else 0)
        })

    category_df = pd.DataFrame(category_data)

    # Table 4: Top Transitions
    pred_transitions = analysis_results['transitions']['Predecessor'].most_common(10)
    rfc_transitions = analysis_results['transitions']['RFC'].most_common(10)

    transitions_data = []
    for i in range(max(len(pred_transitions), len(rfc_transitions))):
        row = {}
        if i < len(pred_transitions):
            row['Predecessor Transition'] = f"{pred_transitions[i][0][0]} → {pred_transitions[i][0][1]}"
            row['Predecessor Count'] = pred_transitions[i][1]
            row['Predecessor %'] = pred_transitions[i][1] / sum(
                analysis_results['transitions']['Predecessor'].values()) * 100 if sum(
                analysis_results['transitions']['Predecessor'].values()) > 0 else 0
        else:
            row['Predecessor Transition'] = ''
            row['Predecessor Count'] = ''
            row['Predecessor %'] = ''

        if i < len(rfc_transitions):
            row['RFC Transition'] = f"{rfc_transitions[i][0][0]} → {rfc_transitions[i][0][1]}"
            row['RFC Count'] = rfc_transitions[i][1]
            row['RFC %'] = rfc_transitions[i][1] / sum(analysis_results['transitions']['RFC'].values()) * 100 if sum(
                analysis_results['transitions']['RFC'].values()) > 0 else 0
        else:
            row['RFC Transition'] = ''
            row['RFC Count'] = ''
            row['RFC %'] = ''

        transitions_data.append(row)

    transitions_df = pd.DataFrame(transitions_data)

    # Table 5: Question Coverage
    coverage_df = pd.DataFrame({
        'Discussion Type': ['Predecessor', 'RFC'],
        'Turns with Questions (%)': [
            analysis_results['question_coverage']['Predecessor'] * 100,
            analysis_results['question_coverage']['RFC'] * 100
        ]
    })

    return {
        'length_stats': length_df,
        'common_flows': common_flows_df,
        'category_counts': category_df,
        'transitions': transitions_df,
        'question_coverage': coverage_df
    }


def main():
    # Load the data
    df = pd.read_csv('results/RFC/results_deepseek_rfc_turn.csv')

    # Get question to category mapping
    question_to_category = get_question_to_category_mapping()

    # Extract flows
    flows = extract_flows(df, question_to_category)

    # Analyze flows
    analysis_results = analyze_flows(flows)

    # Format results into tables
    tables = format_results(analysis_results)

    # Display or save the tables
    print("Flow Length Statistics:")
    print(tables['length_stats'].to_string(index=False))
    print("\nMost Common Flows:")
    print(tables['common_flows'].to_string(index=False))
    print("\nCategory Prevalence:")
    print(tables['category_counts'].to_string(index=False))
    print("\nTop Transitions:")
    print(tables['transitions'].to_string(index=False))
    print("\nQuestion Coverage:")
    print(tables['question_coverage'].to_string(index=False))

    # Save tables to CSV
    tables['length_stats'].to_csv('flow_length_stats.csv', index=False)
    tables['common_flows'].to_csv('common_flows.csv', index=False)
    tables['category_counts'].to_csv('category_prevalence.csv', index=False)
    tables['transitions'].to_csv('top_transitions.csv', index=False)
    tables['question_coverage'].to_csv('question_coverage.csv', index=False)


if __name__ == "__main__":
    main()
