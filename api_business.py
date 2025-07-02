import pandas as pd
import requests
from tqdm import tqdm  # for progress bar
from collections import defaultdict

INPUT_FILE = 'reports/Average price and mix analysis/Average price and mix analysis.xlsx'
OUTPUT_FILE = 'output.xlsx'
API_TOKEN = 'my_token_here'

TEST_MODE = True
MAX_TEST_INNS = 100
TARGET_YEARS = [2022, 2023, 2024, 2025]

def safe_max(current, new):
    if current is None:
        return new
    if new is None:
        return current
    return max(current, new)


def process_tax_data(api_data):
    """Process tax payments and arrears data with null safety"""
    tax_data = defaultdict(lambda: {
        'tax_paid': None,
        'arrears': None,
        'penalties': None,
        'fines': None,
        'total_debt': None
    })

    # Process tax payments
    for payment in api_data.get('taxpay', []):
        year = payment.get('year')
        if year in TARGET_YEARS:
            tax_sum = payment.get('taxsum')
            if tax_sum is not None:
                tax_data[year]['tax_paid'] = safe_max(tax_data[year]['tax_paid'], float(tax_sum))

    # Process arrears
    for arrear in api_data.get('arrear', []):
        year = arrear.get('year')
        if year in TARGET_YEARS:
            fields = {
                'arrears': arrear.get('arrearsum'),
                'penalties': arrear.get('penaltysum'),
                'fines': arrear.get('finesum'),
                'total_debt': arrear.get('totalsum')
            }
            for key, value in fields.items():
                if value is not None:
                    tax_data[year][key] = safe_max(tax_data[year][key], float(value))

    return tax_data


def process_inns():
    print("Reading and preprocessing input file...")
    raw_df = pd.read_excel(INPUT_FILE, sheet_name='data')
    df = raw_df.drop_duplicates(subset=['INN'], keep='first')

    if TEST_MODE:
        df = df.head(MAX_TEST_INNS)
        print(f"\nTEST MODE: Processing only first {MAX_TEST_INNS} INNs")

    results = []

    print("\nQuerying API for each INN...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        inn = str(row['INN']).strip()
        client_name = row['Client name']
        is_ip = any(keyword in client_name.lower()
                    for keyword in ['ип', 'индивидуальный предприниматель'])

        try:
            endpoint = 'search_ip' if is_ip else 'search_org'
            url = f"https://api-cloud.ru/api/pb_nalog.php?type={endpoint}&inn={inn}&token={API_TOKEN}"
            response = requests.get(url, timeout=20)
            search_data = response.json()

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
                            f'{year}_TaxPaid': tax_data[year]['tax_paid'] if tax_data[year][
                                                                                 'tax_paid'] is not None else -1,
                            f'{year}_Arrears': tax_data[year]['arrears'] if tax_data[year][
                                                                                'arrears'] is not None else -1,
                            f'{year}_Penalties': tax_data[year]['penalties'] if tax_data[year][
                                                                                    'penalties'] is not None else -1,
                            f'{year}_Fines': tax_data[year]['fines'] if tax_data[year]['fines'] is not None else -1,
                            f'{year}_TotalDebt': tax_data[year]['total_debt'] if tax_data[year][
                                                                                     'total_debt'] is not None else -1
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

        except Exception as e:
            print(f"\nError processing INN {inn}: {str(e)}")
            results.append({
                'INN': inn,
                'Original Client Name': client_name,
                'Full Name': 'ERROR',
                'Entity Type': 'ИП' if is_ip else 'Юр. лицо',
                'API Response': f'Error: {str(e)}'
            })

    # Create and save output
    print("\nCreating output file...")
    output_df = pd.DataFrame(results)

    # Add original count information
    inn_counts = raw_df['INN'].value_counts().reset_index()
    inn_counts.columns = ['INN', 'CountInOriginalFile']
    output_df = pd.merge(output_df, inn_counts, on='INN', how='left')

    # Reorder columns for better readability
    base_columns = ['INN', 'Original Client Name', 'Full Name', 'Entity Type',
                    'Date Registered', 'Founders Count', 'Total Founder Companies',
                    'Founders List', 'CountInOriginalFile', 'API Response']

    tax_columns = []
    for year in TARGET_YEARS:
        tax_columns.extend([
            f'{year}_TaxPaid',
            f'{year}_Arrears',
            f'{year}_Penalties',
            f'{year}_Fines',
            f'{year}_TotalDebt'
        ])

    output_df = output_df[base_columns + tax_columns]
    output_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Successfully saved results to {OUTPUT_FILE}")

    # Print sample
    print("\nSample output:")
    print(output_df.head().to_string(index=False))


if __name__ == '__main__':
    process_inns()