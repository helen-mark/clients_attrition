import pandas as pd
import numpy as np
import json


def flatten_problem_credit(data):
    flattened = {}

    # Process each INN
    for inn, inn_data in data.items():
        inn_data = inn_data['response'][inn]
        if '_problemCredit' not in inn_data:
            print(1)
            continue

        problem_credit = inn_data['_problemCredit']
        inn_flattened = {}

        # Process each year
        for year, year_data in problem_credit.items():
            # Process top-level metrics (РискЗнач, РискЗона, etc.)
            for metric, value in year_data.items():
                if metric == 'Показатели':
                    continue
                col_name = f'_problemCredit_{metric}_{year}'
                inn_flattened[col_name] = value

            # Process Показатели
            if 'Показатели' in year_data:
                for indicator in year_data['Показатели']:
                    name = indicator['Наименование']
                    for k, v in indicator.items():
                        if k != 'Наименование':
                            col_name = f'{name}_{k}_{year}'
                            inn_flattened[col_name] = v

        flattened[inn] = inn_flattened

    # Create DataFrame
    df = pd.DataFrame.from_dict(flattened, orient='index')

    # Fill missing values with 1000000
    df = df.fillna(1000000)

    return df


def transform_credit_columns(base_df, problem_credit_df):
    """
    Transform year-specific Credit columns into _before and _after columns based on upper_bound date.

    1. For each INN, identify relevant years (upper_bound_year and upper_bound_year - 1)
    2. Create _before and _after columns that pull values from the appropriate year columns
    3. Drop original year-specific Credit columns
    4. Keep all other columns unchanged
    """

    # Make copies to avoid modifying original DataFrames
    base_df = base_df.copy()
    problem_credit_df = problem_credit_df.copy()

    # Ensure INN is the index in both DataFrames
    #if 'INN' in base_df.columns:
    #    base_df = base_df.set_index('INN')
    if 'INN' not in problem_credit_df.columns:
        problem_credit_df = problem_credit_df.reset_index().rename(columns={'index': 'INN'})

    credit_columns = [col for col in problem_credit_df.columns if 'Credit' in col]
    problem_credit_df = problem_credit_df[credit_columns].copy()

    # Merge DataFrames
    print('left:', problem_credit_df)
    print('right:', base_df)
    merged_df = base_df.join(problem_credit_df, how='left').fillna(1000000)
    print('merged:', merged_df)

    # Extract year from upper_bound
    merged_df['_upper_bound_year'] = pd.to_datetime(merged_df['upper_bound']).dt.year

    # Get all Credit columns with years

    # Extract unique metric names and years
    metric_years = {}
    for col in credit_columns:
        print(col)
        # Split column name to extract metric and year
        parts = col.split('_')
        year = None
        for part in parts:
            if len(part) == 4 and part.isdigit():  # Find the year part
                year = int(part)
                break

        if year is not None:
            metric = '_'.join([p for p in parts if not p.isdigit() or len(p) != 4])
            if metric not in metric_years:
                metric_years[metric] = []
            metric_years[metric].append(year)

    # Create before/after columns for each metric
    for metric in metric_years:
        print('next metric...')
        years = sorted(metric_years[metric])

        # Create temporary columns for each possible year
        for year in years:
            col_name = f"{metric}_{year}"
            if col_name not in merged_df.columns:
                merged_df[col_name] = 1000000  # Default value if column doesn't exist

        # Function to get appropriate before/after values for each row
        def get_value(row, time_period):
            current_year = row['_upper_bound_year']
            target_year = current_year - 1

            if current_year in years:
                year_after = current_year
                if current_year - 1 in years:
                    year_before = current_year - 1
                elif current_year - 2 in years:
                    year_before = current_year - 2
                else:
                    year_before = None
            elif current_year - 1 in years:
                year_after = current_year - 1
                if current_year - 2 in years:
                    year_before = current_year - 2
                else:
                    year_before = None

            if year_after is None:
                return 1000000

            if year_after != 2024:
                print('year_after', year_after, year_before)
            closest_year = year_before if time_period == 'before' else year_after
            return row[f"{metric}_{closest_year}"]

        # Create new columns
        merged_df[f"{metric}_before"] = merged_df.apply(lambda x: get_value(x, 'before'), axis=1)
        merged_df[f"{metric}_after"] = merged_df.apply(lambda x: get_value(x, 'after'), axis=1)

    # Drop temporary columns
    merged_df = merged_df.drop(columns=['_upper_bound_year']+credit_columns)
    print(merged_df.columns)

    return merged_df.reset_index()

with open('data/responses_problemCredit.json') as file:
    data = json.load(file)

    df = flatten_problem_credit(data)
    print(df)

    base_data = pd.read_csv('data/v11/train2023-12-01 00:00:00.csv')
    df = transform_credit_columns(base_data, df)
    print(df)
    df.to_csv('dataset_train23.csv')