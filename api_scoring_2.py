import pandas as pd
import requests
from tqdm import tqdm  # for progress bar
from collections import defaultdict
import json
from datetime import datetime

INPUT_FILE_1 = 'reports/Клиенты с датой контрактов.xlsx'
API_TOKEN = 'fd29577658531519f07176e3797adfac80638011'
RESPONSES_FILE = 'data/responses_problemCredit.json'

TEST_MODE = False
MAX_TEST_INNS = 2
TARGET_YEARS = [2021, 2022, 2023, 2024, 2025]

FIN_URL = "https://damia.ru/api-scoring/fincoefs"
BASE_URL = "https://damia.ru/api-scoring/score"


def modify_inn(inn):
    try:
        return float(str(inn).strip().split('.')[0].split(' ')[0])
    except (ValueError, AttributeError):
        # Handle cases where conversion fails (e.g., NaN, non-numeric strings)
        return None

def save_api_response(inn, response_data, is_ip):
    """Save API response to JSON file"""
    try:
        # Try to read existing data
        try:
            with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
                all_responses = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_responses = {}

        # Add new response
        all_responses[inn] = {
            'timestamp': datetime.now().isoformat(),
            'is_ip': is_ip,
            'response': response_data
        }

        # Save back to file
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            print(f'Dumping to {RESPONSES_FILE}...')
            json.dump(all_responses, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error saving response for INN {inn}: {str(e)}")

def get_company_fin_coef(inn):
    params = {
        "inn": inn,
        "key": 'fd29577658531519f07176e3797adfac80638011'
    }

    # try:
    response = requests.get(FIN_URL, params=params)
    print(response)
    print(response.json())

    # Если API возвращает JSON
    return response.json()

def get_company_score(inn, model="_bankrots2016"):
    params = {
        "inn": inn,
        "model": "_problemCredit",
        "key": 'fd29577658531519f07176e3797adfac80638011'
    }

    # try:
    response = requests.get(BASE_URL, params=params)
    print(response)
    print(response.json())

    # Если API возвращает JSON
    return response.json()

def process_inns():
    print("Reading and preprocessing input file...")
    dfs = []
    for year in ['2022', '2023', '2024', '2025']:
        df_1 = pd.read_excel(INPUT_FILE_1, sheet_name=year)
        df_1 = df_1.rename(columns={'ИНН': 'INN', 'НАИМЕНОВАНИЕ КЛИЕНТА': 'Client name'})
        dfs.append(df_1)
    df_1 = pd.concat(dfs)
    df_1['INN'] = df_1['INN'].apply(modify_inn)
    df_1 = df_1.dropna()

    print(f'Total non-unique INNs for Clients with contracts, 2022-2025: {len(df_1)}')
    df = df_1.drop_duplicates(subset=['INN'], keep='first')
    print(f'Total INNs for Clients with contracts, 2022-2025: {len(df)}')


    # df = pd.read_excel(INPUT_FILE, sheet_name='Sheet2')
    #raw_df['INN'] = raw_df['INN'].apply(modify_inn).dropna().replace('inf', 0).astype(int)

    #raw_df = raw_df.drop_duplicates(subset=['INN'], keep='first')
    #print(f'Total INNs for all cities, 2022-2025: {len(raw_df)}')
    #df = raw_df[raw_df['City'] == 'Москва']
    print(f'Total INNs for Moscow, 2022-2025: {len(df)}')
    #df.to_excel('data/unique_inn_moscow.xlsx')

    if TEST_MODE:
        df = df.head(MAX_TEST_INNS)
        print(f"\nTEST MODE: Processing only first {MAX_TEST_INNS} INNs")


    print("\nQuerying API for each INN...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        inn = str(int(row['INN'])).strip()
        #print(inn)
        is_ip = len(inn) > 10 #any(keyword in client_name.lower()
                #    for keyword in ['ип', 'индивидуальный предприниматель'])

        try:
            # endpoint = 'search_ip' if is_ip else 'search_org'
            print(f'INN: {inn}, is ip: {is_ip}')

            response_json = get_company_score(inn)
            save_api_response(inn, response_json, is_ip)

        except Exception as e:
             print(f"\nError processing INN {inn}: {str(e)}")



if __name__ == '__main__':
    process_inns()