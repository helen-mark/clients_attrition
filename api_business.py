import pandas as pd
import requests
from tqdm import tqdm  # for progress bar
from collections import defaultdict
import json
from datetime import datetime

INPUT_FILE_1 = 'reports/Клиенты с датой контрактов.xlsx'
INPUT_FILE = 'output_try1.xlsx'
OUTPUT_FILE = 'output.xlsx'
API_TOKEN = 'e9d8965a9f9089deb9e07f08875fc12f'
RESPONSES_FILE = 'data/responses.json'

TEST_MODE = False
MAX_TEST_INNS = 2
TARGET_YEARS = [2022, 2023, 2024, 2025]

def modify_inn(inn):
    try:
        return float(str(inn).strip().split('.')[0].split(' ')[0])
    except (ValueError, AttributeError):
        # Handle cases where conversion fails (e.g., NaN, non-numeric strings)
        return None

def safe_max(current, new):
    if current is None:
        return new
    if new is None:
        return current
    return max(current, new)


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


def process_tax_data(api_data):
    """Process tax payments and arrears data with null safety"""
    tax_data = defaultdict(lambda: {
        'tax_paid': 0,
        'arrears': 0,
        'penalties': 0,
        'fines': 0,
        'total_debt': 0,
        'kbk_arrear': None,
        'kbkname_arrear': None,
        'kbk_tax': None,
        'kbkname_tax': None,
        'taxCode': None
    })

    # Process tax payments
    for payment in api_data.get('taxpay', []):
        year = payment.get('year')
        if year in TARGET_YEARS:
            tax_sum = payment.get('taxsum')
            if tax_sum is not None:
                tax_data[year]['tax_paid'] += float(tax_sum)
            kbk = payment.get('kbk')
            if kbk is not None:
                tax_data[year]['kbk_tax'] = kbk
            kbkname = payment.get('kbkname')
            if kbkname is not None:
                tax_data[year]['kbkname_tax'] = kbkname


    # Process tax payments
    for taxlist in api_data.get('taxlist', []):
        year = taxlist.get('year')
        if year in TARGET_YEARS:
            taxCode = taxlist.get('taxCode')
            if taxCode is not None:
                tax_data[year]['taxCode'] = taxCode


    # Process arrears
    for arrear in api_data.get('arrear', []):
        year = arrear.get('year')
        if year in TARGET_YEARS:
            a = (arrear.get('arrearsum', 0))
            p = (arrear.get('penaltysum', 0))
            f = (arrear.get('finesum', 0))
            t = (arrear.get('totalsum', 0))

            if a is not None:
                tax_data[year]['arrears'] += float(a)
            if p is not None:
                tax_data[year]['penalties'] += float(p)
            if f is not None:
                tax_data[year]['fines'] += float(f)
            if t is not None:
                tax_data[year]['total_debt'] += float(t)
            #     'kbk_arrear': arrear.get('kbk'),
            #     'kbkname_arrear': arrear.get('kbkname'),
            # }


    return tax_data


def process_inns():
    print("Reading and preprocessing input file...")
    # dfs = []
    # for year in ['2022', '2023', '2024', '2025']:
    #     df_1 = pd.read_excel(INPUT_FILE_1, sheet_name=year)
    #     df_1 = df_1.rename(columns={'ИНН': 'INN', 'НАИМЕНОВАНИЕ КЛИЕНТА': 'Client name'})
    #     dfs.append(df_1)
    # df_1 = pd.concat(dfs)
    # df_1['INN'] = df_1['INN'].apply(modify_inn)
    # print(f'Total non-unique INNs for Clients with contracts, 2022-2025: {len(df_1)}')
    # df_1 = df_1.drop_duplicates(subset=['INN'], keep='first')
    # print(f'Total INNs for Clients with contracts, 2022-2025: {len(df_1)}')


    df = pd.read_excel(INPUT_FILE, sheet_name='Sheet2')
    #raw_df['INN'] = raw_df['INN'].apply(modify_inn).dropna().replace('inf', 0).astype(int)

    #raw_df = raw_df.drop_duplicates(subset=['INN'], keep='first')
    #print(f'Total INNs for all cities, 2022-2025: {len(raw_df)}')
    #df = raw_df[raw_df['City'] == 'Москва']
    print(f'Total INNs for Moscow, 2022-2025: {len(df)}')
    #df.to_excel('data/unique_inn_moscow.xlsx')

    if TEST_MODE:
        df = df.head(MAX_TEST_INNS)
        print(f"\nTEST MODE: Processing only first {MAX_TEST_INNS} INNs")

    results = []

    print("\nQuerying API for each INN...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        inn = str(row['INN']).strip()
        client_name = row['Client name']
        is_ip = len(inn) > 10 #any(keyword in client_name.lower()
                #    for keyword in ['ип', 'индивидуальный предприниматель'])
        print(f'INN: {inn}, is ip: {is_ip}')

        #try:

        endpoint = 'search_ip' if is_ip else 'search_org'
        url = f"https://api-cloud.ru/api/pb_nalog.php?type={endpoint}&inn={inn}&token={API_TOKEN}"
        response = requests.get(url, timeout=200)
        search_data = response.json()
        save_api_response(inn, search_data, is_ip)

        if search_data.get('found') and search_data.get('data'):
            first_result = search_data['data'][0]

            # Base info
            result = {
                'INN': inn,
                'Original Client Name': client_name,
                'Full Name': first_result.get('name', ''),
                'Date Registered': first_result.get('dateReg', ''),
                'Entity Type': 'ИП' if is_ip else 'Юр. лицо',
                'API Response': 'Success'
            }

            # For non-IP entities
            if not is_ip:
                # Founders info
                founders = first_result.get('masuchr', [])
                result.update({
                    'Founders Count': len(founders),
                    'Total Founder Companies': sum(int(f.get('cnt', 0)) for f in founders),
                    'Founders List': "; ".join(f"{f['name']} (ИНН: {f.get('inn', '?')})"
                                               for f in founders) if founders else ""
                })

                # Process tax data with null safety
                tax_data = process_tax_data(first_result)

                # Add tax columns for each year
                for year in TARGET_YEARS:
                    result.update({
                        f'{year}_TaxPaid': tax_data[year]['tax_paid'],
                        f'{year}_Arrears': tax_data[year]['arrears'],
                        f'{year}_Penalties': tax_data[year]['penalties'],
                        f'{year}_Fines': tax_data[year]['fines'],
                        f'{year}_TotalDebt': tax_data[year]['total_debt'],
                        f'{year}_kbk_tax': tax_data[year]['kbk_tax'],
                        f'{year}_kbkname_tax': tax_data[year]['kbkname_tax'],
                        f'{year}_kbk_arrear': tax_data[year]['kbk_arrear'],
                        f'{year}_kbkname_arrear': tax_data[year]['kbkname_arrear'],
                        f'{year}_tax_code': tax_data[year]['taxCode']

                    })

            results.append(result)
        else:
            results.append({
                'INN': inn,
                'Original Client Name': client_name,
                'Full Name': 'NOT FOUND',
                'Entity Type': 'ИП' if is_ip else 'Юр. лицо',
                'API Response': 'Not found'
            })

        # except Exception as e:
        #     print(f"\nError processing INN {inn}: {str(e)}")
        #     results.append({
        #         'INN': inn,
        #         'Original Client Name': client_name,
        #         'Full Name': 'ERROR',
        #         'Entity Type': 'ИП' if is_ip else 'Юр. лицо',
        #         'API Response': f'Error: {str(e)}'
        #     })

    # Create and save output
    print("\nCreating output file...")
    output_df = pd.DataFrame(results)

    # # Add original count information
    # inn_counts = raw_df['INN'].value_counts().reset_index()
    # inn_counts.columns = ['INN', 'CountInOriginalFile']
    # output_df = pd.merge(output_df, inn_counts, on='INN', how='left')

    # Reorder columns for better readability
    base_columns = ['INN', 'Original Client Name', 'Full Name', 'Entity Type',
                    'Date Registered', 'Founders Count', 'Total Founder Companies',
                    'Founders List', 'API Response']

    tax_columns = []
    for year in TARGET_YEARS:
        tax_columns.extend([
            f'{year}_TaxPaid',
            f'{year}_Arrears',
            f'{year}_Penalties',
            f'{year}_Fines',
            f'{year}_TotalDebt',
            f'{year}_kbk_tax',
            f'{year}_kbkname_tax',
            f'{year}_tax_code'
        ])

    output_df = output_df[base_columns + tax_columns]
    output_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Successfully saved results to {OUTPUT_FILE}")

    # Print sample
    print("\nSample output:")
    print(output_df.head().to_string(index=False))


if __name__ == '__main__':
    process_inns()