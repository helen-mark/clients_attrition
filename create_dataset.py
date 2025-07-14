import pandas as pd
import numpy as np
import os
import re
import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import csv
from io import StringIO

cutoff_date = datetime(2025, 2, 10)
cutoff_date_soft = datetime(2025, 5, 10)

class CreateDataset:
    def __init__(self, data_begin_date, data_load_date):
        self.data_begin_date = data_begin_date
        self.data_load_date = data_load_date

        self.renaming_df = pd.read_csv('reports/Переименования/rename_report.csv', delimiter=';', quotechar='"')
        self.renaming_df['OLD_INN'] = self.renaming_df['OLD_INN'].apply(self.modify_inn)
        self.renaming_df['NEW_INN'] = self.renaming_df['NEW_INN'].apply(self.modify_inn)

    def modify_inn(self, inn):
        try:
            return float(str(inn).strip().split('.')[0].split(' ')[0])
        except (ValueError, AttributeError):
            # Handle cases where conversion fails (e.g., NaN, non-numeric strings)
            return None

    def get_main_df(self):
        df = pd.read_excel('reports/Average price and mix analysis/Average price and mix analysis.xlsx', sheet_name='data')
        df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
        print(f'Len of main df: {len(df)}')
        # df = df.loc[df['Month'] < data_load_date]
        # print(f'Len of main df after filter: {len(df)}')

        # df = df[(df['Month'] >= pd.Timestamp('2022-01-01')) | (df['Month'].isna())]
        df['INN'] = df['INN'].apply(self.modify_inn)
        df['INN'] = df['INN'].apply(self.match_inn)

        # extract legal type
        pattern = r'(?<!\S)(ООО|ЗАО|АО|ИП|ПАО|НКО|ОАО|ТОО|ОДО|ЧП)(?!\S)'
        df['legal_type'] = df['Client name'].str.extract(pattern)[0].str.upper()
        df['legal_type'] = df['legal_type'].fillna('Other')

        df = df.sort_values(['INN', 'Month'], ascending=[True, False])
        return df

    def get_recalcs_before_upperbound(self):
        data_path = 'reports/Перерасчеты/'
        dfs = []
        for filename in os.listdir(data_path):
            recalc_df = pd.read_csv(os.path.join(data_path, filename), sep=';',
                                    parse_dates=['ACTION_DATE'], decimal=',',
                                    na_values=['', 'NA', 'N/A', 'NaN', 'nan'])

            # Parse datetime flexibly then keep only date part
            recalc_df['ACTION_DATE'] = pd.to_datetime(
                recalc_df['ACTION_DATE'],
                format='mixed',  # Handles both formats
                dayfirst=True  # Important for European date format
            ).dt.date  # Keep only date part

            # Convert back to datetime (now date-only)
            recalc_df['ACTION_DATE'] = pd.to_datetime(recalc_df['ACTION_DATE'])

            recalc_df = recalc_df.loc[recalc_df['ACTION_DATE'] <= self.data_load_date]
            recalc_df['INN'] = recalc_df['INN'].apply(self.modify_inn).apply(self.match_inn)
            recalc_df = recalc_df[['INN', 'ACTION_DATE', 'SUMMA_DELTA']].copy()
            dfs.append(recalc_df)
        recalc_df = pd.concat(dfs)
        return recalc_df

    def get_contracts_first_dates(self):
        contracts_path = 'reports/Клиенты с датой контрактов.xlsx'
        contracts_df = pd.read_excel(contracts_path, sheet_name='2022')
        contracts_df['ИНН'] = contracts_df['ИНН'].apply(self.modify_inn).apply(self.match_inn)

        for year in ['2023', '2024', '2025']:
            contracts_df_next = pd.read_excel(contracts_path, sheet_name=year)
            contracts_df_next['ИНН'] = contracts_df_next['ИНН'].apply(self.modify_inn).apply(self.match_inn)
            contracts_df = pd.concat([contracts_df, contracts_df_next])

        contracts_df = contracts_df.rename(columns={'ИНН': 'INN'})
        contracts_df['ДАТА КОНТРАКТА'] = pd.to_datetime(contracts_df['ДАТА КОНТРАКТА'])
        contracts_df = contracts_df.groupby('INN')['ДАТА КОНТРАКТА'].min().reset_index()
        return contracts_df

    def add_clear_business(self, result):
        clear_business_df = pd.read_excel('clear_business.xlsx')
        clear_business_df = clear_business_df.drop(columns=['Original Client Name', 'Full Name', 'Founders List', 'API Response'])
        result = pd.merge(result, clear_business_df, on='INN', how='inner')
        result['Date Registered'] = pd.to_datetime(result['Date Registered'], dayfirst=True)
        result['Firm_age_months'] = (result['upper_bound'] - result['Date Registered']).dt.days / 30
        result = result.drop(columns=['Date Registered'])

        # collect taxes, arrears and other info of the previous year for each INN:
        result['ref_year'] = result['upper_bound'].dt.year - 1

        # Step 2: Get all dynamic column suffixes
        year_columns = [col for col in result.columns if col.split('_')[0].isdigit()]
        suffixes = list(set([col.split('_', 1)[1] for col in year_columns]))

        # Step 3: Create new columns (taxes, arrears) based on ref_year
        for suffix in suffixes:
            result[suffix] = result.apply(
                lambda row: row.get(f"{row['ref_year']}_{suffix}", None),
                axis=1
            )
            result[suffix] = result[suffix].replace([np.nan, pd.NA], 0)

        # Step 4: Drop old columns
        result = result.drop(columns=year_columns + ['ref_year'])

        return result

    def do_all_filtering(self, result):
        print(f'result len: {len(result)}')
        result = result.loc[result['Latest_date'] >= self.data_begin_date]
        print(f'result len after filter 1: {len(result)}')
        result = result.loc[(result['ACTIVITY_AND_ATTRITION'] == 0) | (result['Latest_date'] < self.data_load_date)]
        print(f'result len after filter 2: {len(result)}')
        result = result.loc[(result['First_date_from_reports'] < self.data_load_date)]
        print(f'result len after filter 3: {len(result)}')
        result = result[result['unique_addresses_last_12'] < 10]
        # result = result.drop(columns=['upper_bound'])
        print(f'result len after filter 4: {len(result)}')

        result = result.replace([np.inf, -np.inf], np.nan).dropna(how='any')
        print(f'result len after filter 5: {len(result)}')

        inns_list = pd.read_csv('reports/control_group_dataset.csv')[
            'INN'].apply(self.modify_inn).apply(self.match_inn).tolist()  # Convert to list

        df_in_list = result[result['INN'].isin(inns_list)]  # Rows with INNs from Excel

        #df_in_list = result[result['ACTIVITY_AND_ATTRITION']==0].sample(n=1000, random_state=42)
        #df_in_list.to_csv('control_group_dataset.csv')
        #result = result[~result.index.isin(df_in_list.index)]
        result = result[~result['INN'].isin(inns_list)]  # Rows with other INNs
        print(f'result len after eliminating control group: {len(result)}')
        return result, df_in_list

    def load_trip_data(self, base_path):
        all_trips = []

        # Годы и русские названия месяцев
        years = ['22', '23', '24', '25']
        months_ru = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                     'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

        stop = False
        for year in years:
                if stop:
                    break
                # if int(year) < data_begin_date.year:
                #     continue
                print(f'Loading {year} year of trips...')

            # for month_n, month in enumerate(months_ru):
            #     print('n ', month_n)
                # if int(year) == data_begin_date.year and month_n < data_begin_date.month - 1:
                #     continue
                print(f'Loading trips {year} ...')
                if int(year) > self.data_load_date.year:  # and month_n == self.data_load_date.month - 1:
                    print('hit the upper bound')
                    stop = True
                    break
                # Поиск файлов с названием месяца (xlsx или csv)

                for file in os.listdir(base_path):
                    try:
                        if file.endswith('.csv') and 'Москва' in file and year in file:  # and month in file:
                            print('found next trip file')
                            # Use Python's CSV module to handle problematic files
                            with open(os.path.join(base_path, file), 'r', encoding='utf-8') as f:
                                # Read the file content
                                content = f.read()
                                # Create a string buffer
                                buffer = StringIO(content)
                                # Use csv reader with semicolon delimiter
                                reader = csv.reader(buffer, delimiter=';', quotechar='"')
                                # Skip the first two rows
                                # for _ in range(2):
                                #    next(reader)
                                # Read the data
                                data = list(reader)

                            # Get header from row 2 (0-based index after skipping 2 rows)
                            if len(data) > 0:
                                header = data[0]
                                rows = data[1:]
                                df = pd.DataFrame(rows, columns=header)
                            else:
                                print(f"Empty file: {file}")
                                continue
                        else:
                            continue
                        # Добавляем столбцы года и месяца
                        df['S_DAY_ACTION'] = pd.to_datetime(df['S_DAY_ACTION'], dayfirst=True)
                        df['S_DAY_ACTION'] = df['S_DAY_ACTION'].apply(lambda x: x.replace(day=1))
                        df['spec_year'] = (df['S_DAY_ACTION'].dt.year)
                        df['spec_month'] = (df['S_DAY_ACTION'].dt.month)
                        #df = df.rename(columns={'S_DAY_ACTION': 'spec_date'})

                        all_trips.append(df)
                    except Exception as e:
                        print(f"Ошибка при обработке файла {file}: {str(e)}")

        if not all_trips:
            return pd.DataFrame()
        trips_df = pd.concat(all_trips, ignore_index=True)
        trips_df['INN'] = trips_df['INN'].apply(self.modify_inn).apply(self.match_inn)
        return trips_df

    def load_specifications_data(self, base_path):
        all_specs = []

        # Годы и русские названия месяцев
        years = ['2022', '2023', '2024', '2025']
        months_ru = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                     'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

        stop = False
        for year in years:
            if stop:
                break
            # if int(year) < data_begin_date.year:
            #     continue
            print(f'Loading {year} year of specifications...')
            year_path = os.path.join(base_path, year)
            if not os.path.exists(year_path):
                continue

            for month_n, month in enumerate(months_ru):
                # if int(year) == data_begin_date.year and month_n < data_begin_date.month - 1:
                #     continue
                print(f'Loading specs {year} {month}...')
                if int(year) == self.data_load_date.year and month_n == self.data_load_date.month - 1:
                    print('hit the upper bound')
                    stop = True
                    break
                # Поиск файлов с названием месяца (xlsx или csv)
                files = glob.glob(os.path.join(year_path, f'*{month}*.*'))

                for file in files:
                    try:
                        # Определяем формат файла и загружаем
                        if file.endswith('.xlsx'):
                            df = pd.read_excel(file, skiprows=3)
                            df = df.drop(columns=['площадь застила за 28 дней, кв. м', 'примечание', 'тип', 'особенности'])
                            df = df.rename(
                                columns={'ИНН': 'INN', 'адрес': 'ADDRESS', 'юр.наименование': 'LOW_ADDRESS', 'признак "Бюджетный" в счетах клиента': 'BUDGET_STR',
                                         'номер договора': 'C_NUMBER', 'поставщик': 'PROVIDER_NAME',  'об индексации': 'IS_AGREE_TO_INDEXATION_STR',
           'маршруты поездок в интервале дат запроса': 'WAYS_FOR_AC', 'производитель ковра':  'CR_NAME',
           'размер ковра': 'CR_SIZE', 'цвет ковров': 'CR_COLOR_SHORT', 'кол-во ковров': 'AC_QUANTITY', 'менеджер': 'MANAGER',
           'стоимость по спецификации за 28 дней, руб': 'AC_MONEY_USD', 'прайлист': 'CONTRACT_IS_ENDURED_STR',
           'число доставок за 28 дней': 'N_FOR_4WEEKS', 'ОТКРЫТО С ': 'OPEN_FROM', 'ОТКРЫТО ДО': 'OPEN_TO', 'ДОСТАВКА С ': 'DOSTAVKA_FROM',
           'ДОСТАВКА ДО': 'DOSTAVKA_TO', 'код спецификации': 'AC_ID'})  # нормализация названия столбца

                        elif file.endswith('.csv'):
                            # Use Python's CSV module to handle problematic files

                            with open(file, 'r', encoding='utf-8') as f:
                                # Read the file content
                                content = f.read()
                                # Create a string buffer
                                buffer = StringIO(content)
                                # Use csv reader with semicolon delimiter
                                reader = csv.reader(buffer, delimiter=';', quotechar='"')
                                # Skip the first two rows
                                #for _ in range(2):
                                #    next(reader)
                                # Read the data
                                data = list(reader)

                            # Get header from row 2 (0-based index after skipping 2 rows)
                            if len(data) > 0:
                                header = data[0]
                                rows = data[1:]
                                df = pd.DataFrame(rows, columns=header)
                            else:
                                print(f"Empty file: {file}")
                                continue
                        else:
                            continue

                        # Добавляем столбцы года и месяца
                        df['spec_year'] = int(year)
                        df['spec_month'] = months_ru.index(month) + 1
                        df['spec_date'] = pd.to_datetime(df['spec_year'].astype(str) + '-' + df['spec_month'].astype(str))

                        all_specs.append(df)
                    except Exception as e:
                        print(f"Ошибка при обработке файла {file}: {str(e)}")

        if not all_specs:
            return pd.DataFrame()
        specs_df = pd.concat(all_specs, ignore_index=True)
        specs_df['INN'] = specs_df['INN'].apply(self.modify_inn).apply(self.match_inn)
        return specs_df


    def match_inn(self, inn):
        try:
            if inn in self.renaming_df['OLD_INN']:
                sample = self.renaming_df.loc[self.renaming_df['OLD_INN'] == inn]
                if len(sample) == 1: # one or several old inns -> one new inn
                    return sample['NEW_INN'].item()
                elif len(sample['NEW_INN'].unique()) == 1:  # in fact, same situation
                    return sample['NEW_INN'].loc[0].item()
                else:
                    return inn  # if one old inn transfers to several new inns, lets use the old
            elif inn in self.renaming_df['NEW_INN']:
                sample = self.renaming_df.loc[self.renaming_df['NEW_INN'] == inn]
                if len(sample) > 1: # several old inns -> one new inn
                    return inn
                else:
                    old_inn = sample['OLD_INN'].item()
                    old_inns_sample = self.renaming_df.loc[self.renaming_df['OLD_INN'] == old_inn]
                    if len(old_inns_sample) > 0:  # one old inn -> several new inns
                        return old_inn
                    else:
                        return inn
            return inn
        except (ValueError, AttributeError):
            return None

    def is_seasonal(self, group):
        # Find all years when client was active
        active_years = group['Month'].dt.year.unique()

        # Check each year in reverse to find the last summer they could have missed
        k = 0
        for year in sorted(active_years, reverse=True):
            k += 1
            yearly_data = group[group['Month'].dt.year == year]

            # Get all months present in this year
            present_months = yearly_data['Month'].dt.month.unique()

            # Check if summer (June-Aug) is missing
            summer_missing = not any(m in [6, 7, 8] for m in present_months)

            # Check if they were active both before and after summer
            has_pre_summer = any(m < 6 for m in present_months)
            has_post_summer = any(m > 8 for m in present_months)

            # If we found a year with missing summer between activity, mark as seasonal
            if summer_missing and has_pre_summer and has_post_summer:
                return 1
            elif k == 2:
                return 0

        # If no such year found
        return 0

    # Функция для проверки активности
    def is_inactive(self, group):
        latest_date = group['Month'].iloc[0]

        if latest_date.month > self.data_load_date.month: # LSTM rule
            return 0

        if latest_date <= cutoff_date:  # gone long enough
            return 1
        elif latest_date <= cutoff_date_soft: # the client disappeared in spring 2025
            if self.is_seasonal(group):  # seasonal client
                return 0  # assume the client is not gone
            else:  # not seasonal - more likely that the client is really gone
                return 1
        else: # client appeared recently - assume not gone
            return 0

    def add_dates_and_bounds(self, _df: pd.DataFrame):
        latest_dates_tmp = _df.groupby('INN')['Month'].max().reset_index()
        months = (self.data_load_date.year - self.data_begin_date.year) * 12 + (self.data_load_date.month - self.data_begin_date.month) + 1
        random_dates = [
            (self.data_load_date - pd.DateOffset(months=1) * np.random.randint(0, months)).replace(day=1)
            for _ in range(len(latest_dates_tmp))
        ]
        random_dates = pd.DatetimeIndex(random_dates)    # Clip the values
        latest_dates_tmp['upper_bound'] = latest_dates_tmp['Month'].clip(upper=random_dates)
        # latest_dates['upper_bound'] = latest_dates['Month'].clip(upper=data_load_date)
        latest_dates_tmp['upper_bound'] = pd.to_datetime(latest_dates_tmp['upper_bound'])
        latest_dates_tmp = latest_dates_tmp.rename(columns = {'Month': 'Latest_date'})

        # Сортируем по ИНН и дате (новые сначала)
        _df.to_excel('data/df_sorted.xlsx')

        return _df.merge(latest_dates_tmp, on='INN')

    # Функция для получения последних 12 месяцев для каждого ИНН
    def get_last_n_months(self, group, n):
        if len(group) == 0:
            return pd.DataFrame()
        latest_date = group['upper_bound'].iloc[0]
        date_12_months_ago = latest_date - relativedelta(months=n-1)
        _12_months = group[group['Month'].between(date_12_months_ago, latest_date)]
        #if len(_12_months['Month']) > 12:
        #    print(f'---------------Attention: 12 months {_12_months}, {latest_date}, {date_12_months_ago}')
        return _12_months


    # Функция для получения месяцев 4-6 (от последней даты)
    def get_months_4_to_6(self, group):
        if len(group) == 0:
            return pd.DataFrame()
        latest_date = group['upper_bound'].iloc[0]
        start_date = latest_date - relativedelta(months=5)
        end_date = latest_date - relativedelta(months=3)
        return group[group['Month'].between(start_date, end_date)]


    def calculate_area(self, cr_size):
        try:
            width, height = map(int, cr_size.split('*'))
            return width * height
        except:
            return 0  # For invalid formats

    def calc_sqm_single_mats_in_active_specs(self, group):
        # Calculate the product sum
        product = (group['area_cm2'] * group['AC_QUANTITY'])
        total_spacetime_area = product * group['N_FOR_4WEEKS']

        return pd.Series({
            'SQM_SINGLE_MATS_IN_ACTIVE_SPECIFICATIONS': product.sum(),
            'total_spacetime_area': total_spacetime_area.sum()
        })

    def calc_frequency(self, group):
        product_sum = (group['N_FOR_4WEEKS']).sum()

        return pd.Series({
            'Frequency_of_changes_sum': product_sum
        })

    def add_inflation(self, result):
        archive = pd.read_excel('data/инфляция_накопительная.xlsx', sheet_name='Sheet3')
        print('Processing inflation...')
        for idx, row in result.iterrows():
            ub = row['upper_bound']
            coef = archive.loc[archive['year'] == ub.year, ub.month]
            for col in result.columns:
                if 'urnover' in col or 'Price' in col or 'price' in col:
                    result.at[idx, col] = result.at[idx, col] * coef
        return result

    def add_weather(self, result):
        temperature = pd.read_excel('data/архив_погоды_москва.xlsx', sheet_name='температура')
        rains = pd.read_excel('data/архив_погоды_москва.xlsx', sheet_name='осадки')
        for n, archive in enumerate([temperature, rains]):
            print('Processing next archive...')
            for idx, row in result.iterrows():
                ub = row['upper_bound']
                temp = archive.loc[archive['year'].isin([ub.year, ub.year-1])]
                temp_sum = 0
                tems_sum_winter = 0
                for m in range(1, ub.month+1):
                    val = temp.loc[(temp['year'] == ub.year), m].item()
                    temp_sum += val
                    if m in [11, 12, 1, 2, 3]:
                        tems_sum_winter += val
                for m in range(ub.month+1, 13):
                    val = temp.loc[(temp['year'] == ub.year-1), m].item()
                    temp_sum += val
                    if m in [11, 12]:
                        tems_sum_winter += val
                result.at[idx, 'weather_sum_' + str(n)] = temp_sum
                result.at[idx, 'weather_avg_' + str(n)] = temp_sum / 12
                result.at[idx, 'weather_winter_sum_' + str(n)] = tems_sum_winter
                result.at[idx, 'weather_winter_avg_' + str(n)] = tems_sum_winter / 5
        return result

    def collect_all_debits(self):
        base_path = 'reports/Задолженности/'
        all_debits = []

        years = ['2022', '2023', '2024', '2025']
        months_ru = ['.01', '.02', '.03', '.04', '.05', '.06',
                     '.07', '.08', '.09', '.10', '.11', '.12']

        stop = False
        for year in years:
            if stop:
                break
            # if int(year) < data_begin_date.year:
            #     continue
            print(f'Loading {year} year of debits...')

            for month_n, month in enumerate(months_ru):
                print('n ', month_n)
                # if int(year) == data_begin_date.year and month_n < data_begin_date.month - 1:
                #     continue
                print(f'Loading debits {year} {month}...')
                if int(year) == self.data_load_date.year and month_n == self.data_load_date.month - 1:
                    print('hit the upper bound')
                    stop = True
                    break

                for file in os.listdir(os.path.join(base_path, year)):
                    if month not in file:
                        continue
                    df = pd.read_excel(os.path.join(base_path, year, file), skiprows=2)
                    df = df.rename(columns=lambda x: 'Период' if 'Период' in x else x)
                    df = df.rename(columns={'ИНН': 'INN', 'Unnamed: 17': 'Всего'})
                    print(df)
                    df = df[['INN', 'Всего', 'Период']]
                        # Добавляем столбцы года и месяца
                    df['spec_year'] = int(year)
                    df['spec_month'] = months_ru.index(month) + 1
                    df['spec_date'] = pd.to_datetime(df['spec_year'].astype(str) + '-' + df['spec_month'].astype(str))

                    all_debits.append(df)


        if not all_debits:
            return pd.DataFrame()
        debits_df = pd.concat(all_debits, ignore_index=True)
        debits_df['INN'] = debits_df['INN'].apply(self.modify_inn).apply(self.match_inn)

        def safe_convert(val):
            try:
                # Handle NaN/None first
                if pd.isna(val) or val == np.NAN or val is None or val == 'NaN':
                    return 0

                # Convert to string and process
                s = str(val).split(',')[0].replace(' ', '')
                return int(s) if s not in ['', '-'] else 0
            except:
                return 0

        debits_df['Период'] = debits_df['Период'].apply(safe_convert)
        debits_df['Всего'] = debits_df['Всего'].apply(safe_convert)

        return debits_df

    def add_debits(self, result: pd.DataFrame):
        all_debits = self.collect_all_debits()
        latest_dates = result[['INN', 'upper_bound', 'Latest_date', 'start_date']]
        d_merged = pd.merge(all_debits, latest_dates, on='INN', how='inner')
        d_last_12 = d_merged[d_merged['spec_date'].between(
            d_merged['start_date'], d_merged['upper_bound'])]

        d_last_12['debit'] = d_last_12['Всего'] - d_last_12['Период']
        print(f'debits last 12: {d_last_12}')

        d_info = d_last_12.groupby('INN').apply(lambda x: pd.Series({
            'sum_debits': x['debit'].astype(int).sum(),
            'n_debits': x['debit'].notna().astype(int).replace(0, np.nan).count()
            # 'total_debit': x['Всего'].astype(int).iloc[0],
            # 'num_debits': len(x)
        })).reset_index()
        result = pd.merge(result, d_info, on='INN', how='left').fillna(0)
        return result

    def create_dataset(self):
        trips_df = self.load_trip_data('reports/Поездки')
        specs_df = self.load_specifications_data('reports/Спецификации')
        contracts_df = self.get_contracts_first_dates()
        recalc_df = self.get_recalcs_before_upperbound()
        df = self.get_main_df()
        df = self.add_dates_and_bounds(df)

        # DEBUG CHECK
        zero_sqm_groups = df.groupby('INN')['sqm'].sum().loc[lambda x: x == 0]
        print(f"Groups with sqm.sum() = 0:\n{zero_sqm_groups}")
        print("NaN in 'sqm':", df['sqm'].isna().sum())
        print("NaN in 'Turnover':", df['Turnover'].isna().sum())

        # Don't random snapshot clients who left:
        activity_and_attrition = df.groupby('INN').apply(lambda x: pd.Series({
            'ACTIVITY_AND_ATTRITION': self.is_inactive(x),
        })).reset_index()
        df = pd.merge(df, activity_and_attrition, on='INN', how='left')
        mask_1 = df['ACTIVITY_AND_ATTRITION'] == 1
        df.loc[mask_1, 'upper_bound'] = df.loc[mask_1, 'Latest_date'].clip(upper=self.data_load_date)
        df['start_date'] = df['upper_bound'].apply(
            lambda x: x - pd.DateOffset(months=11))

        # Группируем по ИНН и применяем функции
        result = df.groupby('INN').apply(lambda x: pd.Series({
            'sqm_sum': self.get_last_n_months(x, 12)['sqm'].sum(),
            'sqm_mean': self.get_last_n_months(x, 12)['sqm'].mean(),
            'sqm_median': self.get_last_n_months(x, 12)['sqm'].median(),
            'city': x['City'].iloc[0] if 'City' in x.columns else None,
            # 'Cluster': x['Cluster'].iloc[0] if 'Cluster' in x.columns else None,
            'Turnover_sum_last_12': self.get_last_n_months(x, 12)['Turnover'].sum(),
            'Turnover_max_last_12': self.get_last_n_months(x, 12)['Turnover'].max(),
            'Turnover_median_last_12': self.get_last_n_months(x, 12)['Turnover'].median() if x[
                                                                                            'Month'].max() > self.data_begin_date and
                                                                                        x[
                                                                                            'Month'].min() < self.data_load_date else 0,
            # 'Turnover_avg_last_3': get_last_3_months(x)['Turnover'].mean(),
            'Turnover_deriv': self.get_last_n_months(x, 12)['Turnover'].max() - self.get_last_n_months(x, 12)['Turnover'].min(),
            'Latest_date': x['Month'].max(),  # .strftime('%b-%y'),
            'First_date_from_reports': x['Month'].min(),  # .strftime('%b-%y'),
            'start_date': x['start_date'].iloc[0],
            'Price': self.get_last_n_months(x, 12)['Turnover'].sum() / self.get_last_n_months(x, 12)['sqm'].sum() if x[
                                                                                                               'Month'].max() > self.data_begin_date and
                                                                                                           x[
                                                                                                               'Month'].min() < self.data_load_date else 0,
            'ACTIVITY_AND_ATTRITION': self.is_inactive(x),
            'Active_months': len(self.get_last_n_months(x, 12)['Month'].unique()),
            'Seasonality': self.is_seasonal(x),
            'upper_bound': x['upper_bound'].iloc[0],
            'legal_type': x['legal_type'].iloc[0],
        })).reset_index()

        result = self.add_debits(result)

        latest_dates = result[['INN', 'upper_bound', 'Latest_date', 'start_date']]
        # Объединяем с данными спецификаций
        specs_merged = pd.merge(specs_df, latest_dates, on='INN')
        specs_last_12 = specs_merged[specs_merged['spec_date'].between(
            specs_merged['start_date'], specs_merged['upper_bound'])]

        # Группируем по ИНН для агрегации
        specs_addresses = specs_last_12.groupby('INN').agg({
            'ADDRESS': lambda x: x.nunique()  # количество уникальных адресов
        }).reset_index()

        specs_addresses.columns = ['INN', 'unique_addresses_last_12']
        print(f'Specs agg: {specs_addresses["unique_addresses_last_12"]}')
        print(f'Trips df: {trips_df}')

        trips_merged = pd.merge(trips_df, latest_dates, on='INN')
        trips_last_12 = trips_merged[trips_merged['S_DAY_ACTION'].between(
            trips_merged['start_date'], trips_merged['upper_bound'])]

        trips_drivers = trips_last_12.groupby('INN').agg({
            'DRIVER_FIO': lambda x: x.nunique()  # количество уникальных
        }).reset_index()
        trips_drivers.columns = ['INN', 'n_drivers_per_12']

        trips_driver_fio = trips_last_12.groupby('INN')['DRIVER_FIO'].apply(
            lambda x: x.value_counts().idxmax()
            ).reset_index()
        trips_driver_fio['DRIVER_FIO'] = trips_driver_fio['DRIVER_FIO'].replace([np.nan, pd.NA], 'Other')

        trips_statuses = (
            trips_last_12.groupby('INN')['S_STATUS']
            .apply(lambda x: (x == 'недоставлен').mean())
            .reset_index()
        )
        trips_statuses.columns = ['INN', 'undelivered_rate']

        def calculate_area(cr_size):
            try:
                width, height = map(int, cr_size.split('*'))
                return width * height
            except:
                return 0  # For invalid formats

        specs_last_12['area_cm2'] = specs_last_12['CR_SIZE'].apply(calculate_area)
        specs_last_12['AC_QUANTITY'] = pd.to_numeric(specs_last_12['AC_QUANTITY'], errors='coerce').fillna(0)
        specs_last_12['N_FOR_4WEEKS'] = pd.to_numeric(specs_last_12['N_FOR_4WEEKS'], errors='coerce').fillna(0)
        specs_last_12['spacetime_area_fraction'] = specs_last_12['area_cm2'] * specs_last_12['AC_QUANTITY'] * specs_last_12[
            'N_FOR_4WEEKS']
        specs_last_12['spacetime_area_fraction'] /= specs_last_12['spacetime_area_fraction'].sum()

        # Apply to each INN group
        result_SQM_SINGLE_MATS = specs_last_12.groupby('INN').apply(self.calc_sqm_single_mats_in_active_specs).reset_index()
        result_frequency = specs_last_12.groupby('INN').apply(self.calc_frequency).reset_index()
        result_weighted_changes = specs_last_12.groupby('INN').apply(
            lambda x: pd.Series({'weighted_changes': (x['spacetime_area_fraction'] * x['N_FOR_4WEEKS']).sum()}))

        if not specs_addresses.empty:
            result = pd.merge(result, specs_addresses, on='INN', how='inner')
            result = pd.merge(result, result_SQM_SINGLE_MATS, on='INN', how='inner')
            result = pd.merge(result, result_frequency, on='INN', how='inner')
            result = pd.merge(result, result_weighted_changes, on='INN', how='inner')
            result = pd.merge(result, trips_statuses, on='INN', how='inner')
            result = pd.merge(result, trips_drivers, on='INN', how='inner')
            result = pd.merge(result, trips_driver_fio, on='INN', how='inner')
        else:
            print('\n!!!!IMPUTING 0 OF UNIQUE ADDRESSES!!!!\n')
            result['unique_addresses_last_12'] = 0

        upper_bounds = result[['INN', 'upper_bound']].copy()
        # latest_dates['Latest_date'] = pd.to_datetime(latest_dates['Latest_date'])

        recalc_df = recalc_df.dropna(subset=['ACTION_DATE', 'SUMMA_DELTA'])
        upper_bounds = upper_bounds.dropna(subset=['upper_bound'])

        merged_df = pd.merge(recalc_df, upper_bounds, on='INN', how='left')
        merged_df = merged_df.dropna(subset=['upper_bound'])
        merged_df['cutoff_date'] = merged_df['upper_bound'].apply(lambda x: x - relativedelta(months=11))

        # Filter for last 12 months of activity
        last_12m_df = merged_df[merged_df['ACTION_DATE'].between(
            merged_df['cutoff_date'],
            merged_df['upper_bound'],
            inclusive='both'
        )]

        # Group by INN to get stats
        merged_df = last_12m_df.groupby('INN').agg(
            total_recalculations=('SUMMA_DELTA', 'count'),
            sum_recalculations=('SUMMA_DELTA', 'sum')
        ).reset_index()

        # Format results
        merged_df['sum_recalculations'] = merged_df['sum_recalculations'].round(2)
        result = pd.merge(result, merged_df, on='INN', how='left')
        result['total_recalculations'] = result['total_recalculations'].fillna(0)
        result['sum_recalculations'] = result['sum_recalculations'].fillna(0)
        result['Frequency_of_changes_sum'] = result['Frequency_of_changes_sum'].replace([np.inf, -np.inf], 100)

        INN_list = contracts_df['INN']
        result = pd.merge(result, contracts_df, on='INN', how='inner')

        result['Dur_months'] = (result['upper_bound'] - result['ДАТА КОНТРАКТА']).dt.days / 30.44

        print(f'result: {result}')
        result = self.add_weather(result)
        result = self.add_clear_business(result)
        result, control_group_set = self.do_all_filtering(result)
        #control_group_set.to_csv('data/control_v9.csv')
        result.to_csv('data/v11/' + str(self.data_load_date) + '.csv', index=False)
        # df_test.to_csv('data/dataset_v2_test.csv', index=False)


def main():
    #for mnth in range(5,6):
    data_begin_date = datetime(2023, 1, 1)
    data_load_date = datetime(2023, 12, 1)  # contracts info for this year will be loaded
    cd = CreateDataset(data_begin_date, data_load_date)
    cd.create_dataset()



if __name__ == '__main__':
    main()