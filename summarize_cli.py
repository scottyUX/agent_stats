#!/usr/bin/env python3
"""
Summarizes an ai_usage_trace.csv file.
"""

import argparse
import csv
from collections import defaultdict

def summarize_csv(csv_path):
    prompts = 0
    tool_calls = 0
    tool_execution_times = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['event_type'] == 'user_prompt':
                prompts += 1
            elif row['event_type'] == 'tool_call':
                tool_calls += 1
            elif row['event_type'] == 'tool_result':
                tool_name = row['tool_name']
                execution_time = float(row['execution_time'])
                tool_execution_times[tool_name].append(execution_time)

    print("Summary of", csv_path)
    print("-----------")
    print("Total prompts:", prompts)
    print("Total tool calls:", tool_calls)
    print("\\nAverage execution time per tool:")
    for tool, times in tool_execution_times.items():
        avg_time = sum(times) / len(times)
        print(f"- {tool}: {avg_time:.3f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to the ai_usage_trace.csv file")
    args = parser.parse_args()
    summarize_csv(args.csv_path)
