import pandas as pd
import json


def transform_data(json_data):
    result = {}

    for inn, inn_data in json_data.items():
        inn_dict = {}
        response = inn_data.get('response', {}).get(inn, {})

        for coef_name, coef_data in response.items():
            for year, year_data in coef_data.items():
                # Filter for years >= 2021
                if int(year) >= 2021:
                    value = year_data.get('НормаСравн', 'Other')
                    inn_dict[f"{coef_name}_{year}"] = value

        if inn_dict:  # Only add if we found data for 2021+
            result[inn] = inn_dict

    return pd.DataFrame.from_dict(result, orient='index')


# Load your JSON data
with open('your_data.json') as f:
    data = json.load(f)

# Create the DataFrame
df = transform_data(data)
