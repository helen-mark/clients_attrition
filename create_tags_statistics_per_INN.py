import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple


# =============================================================================
# Маппинг контрагентов (INN <-> email <-> phone)
# =============================================================================

class ContactsMapper:
    """
    Загружает merged.csv с колонками INN, phone, email.
    Значения phone/email могут быть перечислены через запятую.
    Строит словари email -> INN и phone -> INN.
    """

    def __init__(
        self,
        mapping_file: str = 'data_cur/mapping/merged.csv',
        inn_col: str = 'INN',
        phone_col: str = 'phone',
        email_col: str = 'email',
    ):
        self.mapping_file = Path(mapping_file)
        self.inn_col = inn_col
        self.phone_col = phone_col
        self.email_col = email_col

        self.email_to_inn: Dict[str, str] = {}
        self.phone_to_inn: Dict[str, str] = {}
        self.stats: Dict[str, int] = {}

    @staticmethod
    def _normalize_inn(x) -> Optional[str]:
        if pd.isna(x):
            return None
        s = str(x).strip().split('.')[0].split(' ')[0]
        return s or None

    @staticmethod
    def _normalize_email(x) -> Optional[str]:
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        return s if '@' in s else None

    @staticmethod
    def _normalize_phone(x) -> Optional[str]:
        """Приводим телефон к 11 цифрам, ведущая 7 или 8 -> 7."""
        if pd.isna(x):
            return None
        digits = re.sub(r'\D', '', str(x))
        if not digits:
            return None
        if len(digits) == 10:          # без кода страны
            digits = '7' + digits
        elif len(digits) == 11 and digits[0] == '8':
            digits = '7' + digits[1:]
        elif len(digits) == 11 and digits[0] == '7':
            pass
        else:
            return None                # не похоже на российский номер
        return digits

    def load(self) -> 'ContactsMapper':
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"Файл маппинга не найден: {self.mapping_file}")

        df = pd.read_csv(self.mapping_file)
        for col in (self.inn_col, self.phone_col, self.email_col):
            if col not in df.columns:
                raise ValueError(f"Колонка '{col}' не найдена в {self.mapping_file}")

        for _, row in df.iterrows():
            inn = self._normalize_inn(row[self.inn_col])
            if not inn:
                continue

            # email может быть списком через запятую
            raw_email = row[self.email_col]
            if not pd.isna(raw_email):
                for raw in str(raw_email).split(','):
                    em = self._normalize_email(raw)
                    if em:
                        self.email_to_inn[em] = inn

            # phone может быть списком через запятую
            raw_phone = row[self.phone_col]
            if not pd.isna(raw_phone):
                for raw in str(raw_phone).split(','):
                    ph = self._normalize_phone(raw)
                    if ph:
                        self.phone_to_inn[ph] = inn

        self.stats['emails_mapped'] = len(self.email_to_inn)
        self.stats['phones_mapped'] = len(self.phone_to_inn)
        print(f"Маппинг: {len(df)} строк -> "
              f"{len(self.email_to_inn)} email, {len(self.phone_to_inn)} телефонов")
        return self


# =============================================================================
# Основной процессор
# =============================================================================

class MailTagProcessor:
    """
    Обработка почтовой истории И телефонных звонков, матчинг по email/phone
    через merged.csv, подсчет общей статистики тегов по ИНН и месяцам.
    """

    def __init__(
        self,
        mapping_file: str = 'data_cur/mapping/merged.csv',
        mail_file: str = 'data_cur/mail_tagged_plus_llm.csv',
        calls_file: str = 'data_cur/calls_plus_llm.csv',
        # --- колонки маппинга ---
        contacts_inn_col: str = 'INN',
        contacts_email_col: str = 'email',
        contacts_phone_col: str = 'phone',
        # --- колонки почты ---
        mail_from_col: str = 'from',
        mail_date_col: str = 'date_str',
        mail_tags_col: str = 'tags',
        # --- колонки звонков ---
        calls_audio_col: str = 'source_audio',
        calls_tags_col: str = 'tags',
        calls_date_col: Optional[str] = None,   # если нет — берём из source_audio
        # --- фильтрация ---
        exclude_tags: Optional[List[str]] = None,
    ):
        self.mapping_file = Path(mapping_file)
        self.mail_file = Path(mail_file)
        self.calls_file = Path(calls_file)

        self.contacts_inn_col = contacts_inn_col
        self.contacts_email_col = contacts_email_col
        self.contacts_phone_col = contacts_phone_col

        self.mail_from_col = mail_from_col
        self.mail_date_col = mail_date_col
        self.mail_tags_col = mail_tags_col

        self.calls_audio_col = calls_audio_col
        self.calls_tags_col = calls_tags_col
        self.calls_date_col = calls_date_col

        self.exclude_tags = exclude_tags if exclude_tags is not None else ['mail', 'ai rct']

        # данные
        self.mapper: Optional[ContactsMapper] = None
        self.df_mail: Optional[pd.DataFrame] = None
        self.df_calls: Optional[pd.DataFrame] = None
        self.df_combined: Optional[pd.DataFrame] = None
        self.df_statistics: Optional[pd.DataFrame] = None

        self.stats: Dict = {}

    # ------------------------------------------------------------------ load
    def load_data(self) -> 'MailTagProcessor':
        print("Загрузка файлов...")

        # 1. маппинг
        self.mapper = ContactsMapper(
            mapping_file=str(self.mapping_file),
            inn_col=self.contacts_inn_col,
            phone_col=self.contacts_phone_col,
            email_col=self.contacts_email_col,
        ).load()
        self.stats.update(self.mapper.stats)

        # 2. почта
        if not self.mail_file.exists():
            raise FileNotFoundError(f"Файл почты не найден: {self.mail_file}")
        self.df_mail = pd.read_csv(self.mail_file)
        self.df_mail['__channel'] = 'mail'
        print(f"Загружено {len(self.df_mail)} писем")

        for col in (self.mail_from_col, self.mail_date_col, self.mail_tags_col):
            if col not in self.df_mail.columns:
                raise ValueError(f"Колонка '{col}' не найдена в файле почты")

        # 3. звонки
        if not self.calls_file.exists():
            raise FileNotFoundError(f"Файл звонков не найден: {self.calls_file}")
        self.df_calls = pd.read_csv(self.calls_file)
        self.df_calls['__channel'] = 'call'
        print(f"Загружено {len(self.df_calls)} звонков")

        if self.calls_audio_col not in self.df_calls.columns:
            raise ValueError(f"Колонка '{self.calls_audio_col}' не найдена в файле звонков")
        if self.calls_tags_col not in self.df_calls.columns:
            raise ValueError(f"Колонка '{self.calls_tags_col}' не найдена в файле звонков")

        return self

    # ------------------------------------------------------------- extraction
    @staticmethod
    def _extract_email(from_string) -> Optional[str]:
        if pd.isna(from_string):
            return None
        match = re.search(r'<(.*?)>', str(from_string))
        if match:
            return match.group(1).strip().lower()
        if '@' in str(from_string):
            return str(from_string).strip().lower()
        return None

    @staticmethod
    def _extract_phone_from_source_audio(source_audio) -> Optional[str]:
        """
        Из имени файла вида
          2026-09-23_15-24-36_79030948738_call_6008207383_mp3_<hash>.mp3
        достаёт номер звонящего.
        """
        if pd.isna(source_audio):
            return None
        name = Path(str(source_audio)).name

        # вариант 1: ..._<phone>_call_...
        m = re.search(r'_(\d{10,15})_call_', name)
        if m:
            return ContactsMapper._normalize_phone(m.group(1))

        # вариант 2: любой длинный номер в имени
        for digits in re.findall(r'\d{10,15}', name):
            ph = ContactsMapper._normalize_phone(digits)
            if ph:
                return ph
        return None

    @staticmethod
    def _extract_datetime_from_source_audio(source_audio) -> Optional[pd.Timestamp]:
        """Из 2026-09-23_15-24-36_... достаёт Timestamp."""
        if pd.isna(source_audio):
            return None
        name = Path(str(source_audio)).name
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', name)
        if not m:
            return None
        y, mo, d, h, mi, s = m.groups()
        try:
            return pd.Timestamp(f"{y}-{mo}-{d} {h}:{mi}:{s}")
        except Exception:
            return None

    def extract_emails(self) -> 'MailTagProcessor':
        print("\nИзвлечение email из писем...")
        self.df_mail['email'] = self.df_mail[self.mail_from_col].apply(self._extract_email)
        self.df_mail['phone'] = None
        extracted = self.df_mail['email'].notna().sum()
        self.stats['emails_extracted'] = int(extracted)
        print(f"  извлечено email: {extracted} из {len(self.df_mail)}")
        return self

    def extract_phones(self) -> 'MailTagProcessor':
        print("\nИзвлечение телефонов из звонков...")
        self.df_calls['phone'] = self.df_calls[self.calls_audio_col].apply(
            self._extract_phone_from_source_audio
        )
        self.df_calls['email'] = None

        # дата: либо своя колонка, либо из source_audio
        if self.calls_date_col and self.calls_date_col in self.df_calls.columns:
            self.df_calls['date'] = pd.to_datetime(
                self.df_calls[self.calls_date_col], format='mixed', errors='coerce'
            )
            mask = self.df_calls['date'].isna()
            self.df_calls.loc[mask, 'date'] = self.df_calls.loc[mask, self.calls_audio_col].apply(
                self._extract_datetime_from_source_audio
            )
        else:
            self.df_calls['date'] = self.df_calls[self.calls_audio_col].apply(
                self._extract_datetime_from_source_audio
            )

        extracted = self.df_calls['phone'].notna().sum()
        dated = self.df_calls['date'].notna().sum()
        self.stats['phones_extracted'] = int(extracted)
        self.stats['calls_with_date'] = int(dated)
        print(f"  извлечено телефонов: {extracted} из {len(self.df_calls)}")
        print(f"  дат извлечено: {dated} из {len(self.df_calls)}")
        return self

    # -------------------------------------------------------------- matching
    def match_inn(self) -> 'MailTagProcessor':
        print("\nМатчинг ИНН...")

        # --- почта: сначала email, потом phone ---
        self.df_mail['ИНН'] = self.df_mail['email'].map(self.mapper.email_to_inn)
        mask = self.df_mail['ИНН'].isna() & self.df_mail['phone'].notna()
        if mask.any():
            self.df_mail.loc[mask, 'ИНН'] = self.df_mail.loc[mask, 'phone'].map(
                self.mapper.phone_to_inn
            )
        mail_matched = int(self.df_mail['ИНН'].notna().sum())
        print(f"  писем сматчено: {mail_matched} из {len(self.df_mail)}")

        # --- звонки: по phone ---
        self.df_calls['ИНН'] = self.df_calls['phone'].map(self.mapper.phone_to_inn)
        calls_matched = int(self.df_calls['ИНН'].notna().sum())
        print(f"  звонков сматчено: {calls_matched} из {len(self.df_calls)}")

        self.stats['mail_matched'] = mail_matched
        self.stats['calls_matched'] = calls_matched

        # оставляем только сматченные
        self.df_mail = self.df_mail.dropna(subset=['ИНН']).copy()
        self.df_calls = self.df_calls.dropna(subset=['ИНН']).copy()
        self.stats['mail_after_match'] = len(self.df_mail)
        self.stats['calls_after_match'] = len(self.df_calls)
        print(f"  после фильтрации: {len(self.df_mail)} писем, {len(self.df_calls)} звонков")
        return self

    # -------------------------------------------------------- combine & clean
    def combine(self) -> 'MailTagProcessor':
        """
        Объединяет письма и звонки в один DataFrame со стандартными колонками:
        date, tags, ИНН, channel, contact.
        """
        print("\nОбъединение писем и звонков...")

        mail = pd.DataFrame({
            'date':      pd.to_datetime(self.df_mail[self.mail_date_col],
                                        format='mixed', errors='coerce'),
            'tags':      self.df_mail[self.mail_tags_col],
            'ИНН':       self.df_mail['ИНН'],
            'channel':   'mail',
            'contact':   self.df_mail['email'],
        })

        calls = pd.DataFrame({
            'date':      pd.to_datetime(self.df_calls['date'], errors='coerce'),
            'tags':      self.df_calls[self.calls_tags_col],
            'ИНН':       self.df_calls['ИНН'],
            'channel':   'call',
            'contact':   self.df_calls['phone'],
        })

        self.df_combined = pd.concat([mail, calls], ignore_index=True)
        self.df_combined = self.df_combined.dropna(subset=['date']).copy()
        print(f"  объединено записей: {len(self.df_combined)}")
        print(self.df_combined['channel'].value_counts().to_string())
        return self

    @staticmethod
    def _parse_tags(tags_string) -> List[str]:
        if pd.isna(tags_string):
            return []
        try:
            tags_str = str(tags_string).strip().strip('[]')
            if not tags_str:
                return []
            return [t.strip().strip("'\"") for t in tags_str.split(',') if t.strip()]
        except Exception:
            return []

    def parse_tags(self) -> 'MailTagProcessor':
        print("\nПарсинг тегов...")
        self.df_combined['tags_list'] = self.df_combined['tags'].apply(self._parse_tags)
        total = int(self.df_combined['tags_list'].apply(len).sum())
        self.stats['total_tags_parsed'] = total
        print(f"  всего тегов: {total}")
        return self

    def prepare_dates(self) -> 'MailTagProcessor':
        print("\nОбработка дат...")
        self.df_combined['date'] = pd.to_datetime(self.df_combined['date'], errors='coerce')
        self.df_combined['year_month'] = self.df_combined['date'].dt.to_period('M')
        self.df_combined = self.df_combined.dropna(subset=['date']).copy()
        self.stats['date_range_start'] = self.df_combined['date'].min()
        self.stats['date_range_end'] = self.df_combined['date'].max()
        print(f"  диапазон: {self.stats['date_range_start'].date()} .. "
              f"{self.stats['date_range_end'].date()}")
        return self

    # ------------------------------------------------------------- statistics
    def calculate_statistics(self) -> 'MailTagProcessor':
        """
        Общая статистика тегов по ИНН и месяцу — БЕЗ разделения на каналы.
        """
        print("\nПодсчет статистики тегов по ИНН и месяцам...")
        exclude = set(self.exclude_tags or [])

        results = []
        for (inn, period), group in self.df_combined.groupby(['ИНН', 'year_month']):
            counter: Dict[str, int] = {}
            for tags in group['tags_list']:
                for tag in tags:
                    if tag in exclude:
                        continue
                    counter[tag] = counter.get(tag, 0) + 1
            for tag, cnt in counter.items():
                results.append({
                    'ИНН': inn,
                    'Год-Месяц': str(period),
                    'Тег': tag,
                    'Количество': cnt,
                })

        if results:
            self.df_statistics = (
                pd.DataFrame(results)
                .sort_values(['ИНН', 'Год-Месяц', 'Тег'])
                .reset_index(drop=True)
            )
            self.stats['total_stats_records'] = len(self.df_statistics)
            self.stats['unique_inn_in_stats'] = self.df_statistics['ИНН'].nunique()
            self.stats['unique_tags_in_stats'] = self.df_statistics['Тег'].nunique()
            print(f"  записей: {len(self.df_statistics)}, "
                  f"ИНН: {self.df_statistics['ИНН'].nunique()}, "
                  f"тегов: {self.df_statistics['Тег'].nunique()}")
        else:
            self.df_statistics = pd.DataFrame(
                columns=['ИНН', 'Год-Месяц', 'Тег', 'Количество']
            )
            print("  статистика пуста")
        return self

    # ----------------------------------------------------------------- output
    def save_results(
        self,
        statistics_file: str = 'tag_statistics_by_inn_monthly.csv',
        intermediate_file: str = 'mail_calls_with_inn.csv',
    ) -> 'MailTagProcessor':
        print("\nСохранение результатов...")
        if self.df_statistics is not None and len(self.df_statistics) > 0:
            self.df_statistics.to_csv(statistics_file, index=False, encoding='utf-8')
            print(f"  статистика: {statistics_file}")
        if self.df_combined is not None:
            self.df_combined.to_csv(intermediate_file, index=False, encoding='utf-8')
            print(f"  промежуточный: {intermediate_file}")
        return self

    def print_summary(self):
        print("\n=== Сводная статистика ===")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        if self.df_statistics is not None and len(self.df_statistics) > 0:
            print("\n=== Топ-10 тегов ===")
            top = (self.df_statistics.groupby('Тег')['Количество'].sum()
                   .sort_values(ascending=False).head(10))
            for tag, cnt in top.items():
                print(f"  {tag}: {cnt}")

    # ------------------------------------------------------------------- run
    def run_pipeline(
        self,
        save: bool = True,
        statistics_file: str = 'tag_statistics_by_inn_monthly.csv',
        intermediate_file: str = 'mail_calls_with_inn.csv',
    ) -> pd.DataFrame:
        (self
         .load_data()
         .extract_emails()
         .extract_phones()
         .match_inn()
         .combine()
         .parse_tags()
         .prepare_dates()
         .calculate_statistics())
        if save:
            self.save_results(statistics_file, intermediate_file)
        self.print_summary()
        return self.df_statistics

    # ------------------------------------------------------------- accessors
    def get_statistics(self) -> pd.DataFrame:
        return self.df_statistics

    def get_matched_data(self) -> pd.DataFrame:
        return self.df_combined

    def get_stats_summary(self) -> Dict:
        return self.stats


# =============================================================================
# Feature engineering
# =============================================================================

class MailTagFeatureEngineer:
    PERIODS = [(1, 'last_1m'), (3, 'last_3m'), (6, 'last_6m'), (12, 'last_12m')]
    ACTIVITY_PERIODS = (3, 6, 12)
    DIVERSITY_PERIODS = (1, 3, 6, 12)
    DERIVATIVE_PAIRS = [
        ('last_1m', 1, 'prev_1m', 2),
        ('last_3m', 3, 'prev_3m', 6),
        ('last_6m', 6, 'prev_6m', 12),
    ]

    def __init__(self, tag_statistics: pd.DataFrame):
        self.tag_stats = tag_statistics.copy()

        # нормализуем ИНН к строке без .0 и пробелов
        if 'ИНН' in self.tag_stats.columns:
            self.tag_stats['ИНН'] = self.tag_stats['ИНН'].apply(self._normalize_inn)
        else:
            raise ValueError("В tag_statistics нет колонки 'ИНН'")

        # нормализуем 'Год-Месяц' -> date
        if 'Год-Месяц' in self.tag_stats.columns:
            ym = self.tag_stats['Год-Месяц'].astype(str).str.strip()
        else:
            raise ValueError("В tag_statistics нет колонки 'Год-Месяц'")
        self.tag_stats['date'] = pd.to_datetime(ym + '-01', errors='coerce')
        self.tag_stats = self.tag_stats.dropna(subset=['date']).copy()

        # нормализуем названия тегов
        self.tag_stats['Тег_norm'] = self.tag_stats['Тег'].apply(self._normalize_tag_name)

        # заранее сгруппируем по ИНН для быстрого доступа
        self._by_inn = {inn: g for inn, g in self.tag_stats.groupby('ИНН')}

        self.all_tags = sorted(self.tag_stats['Тег_norm'].unique())
        print(f"[engineer] уникальных тегов: {len(self.all_tags)}; "
              f"уникальных ИНН: {self.tag_stats['ИНН'].nunique()}")

    @staticmethod
    def _normalize_inn(x) -> Optional[str]:
        if pd.isna(x):
            return None
        s = str(x).strip().split('.')[0].split(' ')[0]
        return s or None

    @staticmethod
    def _normalize_tag_name(tag: str) -> str:
        s = str(tag).strip().lower()
        s = re.sub(r'[^\w]+', '_', s, flags=re.UNICODE).strip('_')
        return s or 'unknown'

    # ------------------------------------------------------------------ schema
    def _all_feature_names(self) -> set:
        """
        Единый источник истины — полный набор имён признаков, которые
        генерирует класс для любого клиента.
        """
        names = set()

        # периоды
        for _, prefix in self.PERIODS:
            names |= {f'{prefix}_total_tags',
                      f'{prefix}_active_months',
                      f'{prefix}_avg_monthly_tags'}
            for tag in self.all_tags:
                names.add(f'{prefix}_{tag}')

        # производные
        for curr, _, prev, _ in self.DERIVATIVE_PAIRS:
            names |= {f'{curr}_vs_{prev}_total_diff',
                      f'{curr}_vs_{prev}_total_pct_change'}
            for tag in self.all_tags:
                names.add(f'{curr}_vs_{prev}_{tag}_diff')

        # тренды
        names |= set(self._zero_trend().keys())

        # активность
        for months in self.ACTIVITY_PERIODS:
            p = f'last_{months}m'
            names |= {f'{p}_density', f'{p}_frequency',
                      f'{p}_has_gaps', f'{p}_max_gap_months'}
        names |= {'activity_decay_3m_vs_6m', 'activity_decay_1m_vs_3m'}

        # разнообразие
        for months in self.DIVERSITY_PERIODS:
            p = f'last_{months}m'
            names |= {f'{p}_unique_tags', f'{p}_tag_diversity',
                      f'{p}_tag_entropy', f'{p}_normalized_entropy'}

        # темпоральные
        names |= set(self._empty_temporal().keys())

        # сезонные
        names |= set(self._empty_seasonal().keys())

        return names

    # -------------------------------------------------------- период-фичи
    def _period_features(self, data, upper_bound, months, prefix) -> Dict:
        start = upper_bound - pd.DateOffset(months=months)
        mask = (data['date'] >= start) & (data['date'] < upper_bound)
        sub = data[mask]

        total = float(sub['Количество'].sum()) if len(sub) else 0.0
        f = {f'{prefix}_total_tags': total}

        if len(sub):
            by_tag = sub.groupby('Тег_norm')['Количество'].sum()
        else:
            by_tag = pd.Series(dtype=float)

        for tag in self.all_tags:
            f[f'{prefix}_{tag}'] = int(by_tag.get(tag, 0))

        active_months = int(sub['Год-Месяц'].nunique()) if len(sub) else 0
        f[f'{prefix}_active_months'] = active_months
        f[f'{prefix}_avg_monthly_tags'] = (
            total / min(months, active_months) if active_months > 0 else 0.0
        )
        return f

    # ------------------------------------------------------------ производные
    def _derivative_features(self, data, upper_bound) -> Dict:
        f = {}
        for curr, cm, prev, pm in self.DERIVATIVE_PAIRS:
            cs = upper_bound - pd.DateOffset(months=cm)
            ps = upper_bound - pd.DateOffset(months=pm)
            cur = data[(data['date'] >= cs) & (data['date'] < upper_bound)]
            prv = data[(data['date'] >= ps) & (data['date'] < cs)]

            cur_total = float(cur['Количество'].sum()) if len(cur) else 0.0
            prv_total = float(prv['Количество'].sum()) if len(prv) else 0.0

            f[f'{curr}_vs_{prev}_total_diff'] = cur_total - prv_total
            f[f'{curr}_vs_{prev}_total_pct_change'] = (
                (cur_total - prv_total) / prv_total * 100 if prv_total > 0
                else (100.0 if cur_total > 0 else 0.0)
            )

            cur_by = (cur.groupby('Тег_norm')['Количество'].sum()
                      if len(cur) else pd.Series(dtype=float))
            prv_by = (prv.groupby('Тег_norm')['Количество'].sum()
                      if len(prv) else pd.Series(dtype=float))
            for tag in self.all_tags:
                f[f'{curr}_vs_{prev}_{tag}_diff'] = int(
                    cur_by.get(tag, 0) - prv_by.get(tag, 0)
                )
        return f

    # ------------------------------------------------------------------ тренды
    def _trend_features(self, data, upper_bound) -> Dict:
        start = upper_bound - pd.DateOffset(months=12)
        sub = data[(data['date'] >= start) & (data['date'] < upper_bound)]
        if len(sub) == 0:
            return self._zero_trend()
        monthly = sub.groupby('date')['Количество'].sum().sort_index()
        if len(monthly) < 2:
            return self._zero_trend()
        x = np.arange(len(monthly))
        y = monthly.values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0])
        mean_y = float(np.mean(y)) if np.mean(y) > 0 else 0.0
        return {
            'trend_slope_12m': slope,
            'trend_direction_12m': 1 if slope > 0 else (-1 if slope < 0 else 0),
            'trend_volatility_12m': float(np.std(y)),
            'trend_cv_12m': float(np.std(y) / mean_y) if mean_y > 0 else 0.0,
            'trend_max_monthly_12m': float(np.max(y)),
            'trend_last_to_max_ratio': float(y[-1] / np.max(y)) if np.max(y) > 0 else 0.0,
        }

    @staticmethod
    def _zero_trend() -> Dict:
        return {
            'trend_slope_12m': 0.0, 'trend_direction_12m': 0,
            'trend_volatility_12m': 0.0, 'trend_cv_12m': 0.0,
            'trend_max_monthly_12m': 0.0, 'trend_last_to_max_ratio': 0.0,
        }

    # -------------------------------------------------------------- активность
    def _activity_features(self, data, upper_bound) -> Dict:
        f = {}
        for months in self.ACTIVITY_PERIODS:
            start = upper_bound - pd.DateOffset(months=months)
            sub = data[(data['date'] >= start) & (data['date'] < upper_bound)]
            prefix = f'last_{months}m'
            if len(sub):
                active = int(sub['Год-Месяц'].nunique())
                total = float(sub['Количество'].sum())
                f[f'{prefix}_density'] = total / active if active else 0.0
                f[f'{prefix}_frequency'] = active / months
                f[f'{prefix}_has_gaps'] = 1 if active < months else 0
                f[f'{prefix}_max_gap_months'] = months - active
            else:
                f[f'{prefix}_density'] = 0.0
                f[f'{prefix}_frequency'] = 0.0
                f[f'{prefix}_has_gaps'] = 1
                f[f'{prefix}_max_gap_months'] = months
        f['activity_decay_3m_vs_6m'] = f['last_3m_frequency'] - f['last_6m_frequency']
#        f['activity_decay_1m_vs_3m'] = f['last_1m_frequency'] - f['last_3m_frequency']
        return f

    # ------------------------------------------------------------ разнообразие
    def _diversity_features(self, data, upper_bound) -> Dict:
        f = {}
        for months in self.DIVERSITY_PERIODS:
            start = upper_bound - pd.DateOffset(months=months)
            sub = data[(data['date'] >= start) & (data['date'] < upper_bound)]
            prefix = f'last_{months}m'
            if len(sub):
                uniq = int(sub['Тег_norm'].nunique())
                f[f'{prefix}_unique_tags'] = uniq
                f[f'{prefix}_tag_diversity'] = uniq / len(self.all_tags) if self.all_tags else 0.0
                counts = sub.groupby('Тег_norm')['Количество'].sum()
                tot = float(counts.sum())
                if tot > 0 and len(counts) > 1:
                    p = counts / tot
                    ent = float(-np.sum(p * np.log2(p)))
                    max_ent = np.log2(len(self.all_tags)) if len(self.all_tags) > 1 else 1.0
                    f[f'{prefix}_tag_entropy'] = ent
                    f[f'{prefix}_normalized_entropy'] = ent / max_ent if max_ent > 0 else 0.0
                else:
                    f[f'{prefix}_tag_entropy'] = 0.0
                    f[f'{prefix}_normalized_entropy'] = 0.0
            else:
                f[f'{prefix}_unique_tags'] = 0
                f[f'{prefix}_tag_diversity'] = 0.0
                f[f'{prefix}_tag_entropy'] = 0.0
                f[f'{prefix}_normalized_entropy'] = 0.0
        return f

    # -------------------------------------------------------------- сезонность
    def _get_seasonal_features(self, data, upper_bound) -> Dict:
        """
        Сезонные признаки. Анализирует последние 24 месяца.
        """
        start = upper_bound - pd.DateOffset(months=24)
        sub = data[(data['date'] >= start) & (data['date'] < upper_bound)].copy()

        if len(sub) == 0:
            return self._empty_seasonal()

        sub['month'] = sub['date'].dt.month
        sub['year'] = sub['date'].dt.year

        last_year_same_month = upper_bound - pd.DateOffset(years=1)

        current_month_data = float(sub[
            (sub['month'] == upper_bound.month) &
            (sub['year'] == upper_bound.year)
        ]['Количество'].sum())

        prev_year_month_data = float(sub[
            (sub['month'] == upper_bound.month) &
            (sub['year'] == last_year_same_month.year)
        ]['Количество'].sum())

        yoy_change = current_month_data - prev_year_month_data
        yoy_pct = (
            (current_month_data - prev_year_month_data) / prev_year_month_data * 100
            if prev_year_month_data > 0
            else (100.0 if current_month_data > 0 else 0.0)
        )

        monthly_avg = sub.groupby('month')['Количество'].mean()
        if len(monthly_avg) > 0:
            median_activity = float(monthly_avg.median())
            is_high_season = (
                1 if (upper_bound.month in monthly_avg.index and
                      monthly_avg.get(upper_bound.month, 0) > median_activity)
                else 0
            )
        else:
            is_high_season = 0

        return {
            'seasonal_same_month_last_year': prev_year_month_data,
            'seasonal_current_month': current_month_data,
            'seasonal_yoy_change': yoy_change,
            'seasonal_yoy_pct_change': yoy_pct,
            'is_high_season': is_high_season,
        }

    @staticmethod
    def _empty_seasonal() -> Dict:
        return {
            'seasonal_same_month_last_year': 0.0,
            'seasonal_current_month': 0.0,
            'seasonal_yoy_change': 0.0,
            'seasonal_yoy_pct_change': 0.0,
            'is_high_season': 0,
        }

    # ------------------------------------------------------------ темпоральные
    def _temporal_features(self, data, upper_bound) -> Dict:
        sub = data[data['date'] < upper_bound]
        if len(sub) == 0:
            return self._empty_temporal()

        f = {}
        last = sub['date'].max()
        days = max(0, (upper_bound - last).days)
        f['days_since_last_communication'] = days
        f['months_since_last_communication'] = days / 30.44
        f['has_communication_last_30d'] = 1 if days <= 30 else 0
        f['has_communication_last_60d'] = 1 if days <= 60 else 0
        f['has_communication_last_90d'] = 1 if days <= 90 else 0

        first = sub['date'].min()
        dur = (upper_bound - first).days
        f['relationship_duration_days'] = dur
        f['relationship_duration_months'] = dur / 30.44
        f['total_active_months'] = int(sub['Год-Месяц'].nunique())

        monthly = sub.groupby('date')['Количество'].sum().sort_index()

        if len(monthly) > 1:
            intervals = monthly.index.to_series().diff().dt.days.dropna()
            f['avg_interval_days'] = float(intervals.mean()) if len(intervals) else 0.0
            f['std_interval_days'] = float(intervals.std()) if len(intervals) > 1 else 0.0
            f['max_interval_days'] = float(intervals.max()) if len(intervals) else 0.0
            f['min_interval_days'] = float(intervals.min()) if len(intervals) else 0.0
            f['days_since_last_vs_avg_interval'] = (
                days / f['avg_interval_days'] if f['avg_interval_days'] > 0 else days
            )
        else:
            f['avg_interval_days'] = 0.0
            f['std_interval_days'] = 0.0
            f['max_interval_days'] = 0.0
            f['min_interval_days'] = 0.0
            f['days_since_last_vs_avg_interval'] = days

        if f['avg_interval_days'] > 0:
            f['is_overdue'] = 1 if days > f['avg_interval_days'] * 1.5 else 0
        else:
            f['is_overdue'] = 0

        if len(monthly) >= 3:
            recent_3 = float(monthly.tail(3).mean())
            older = monthly.iloc[:-3]
            older_avg = float(older.mean()) if len(older) > 0 else 0.0
            f['activity_ratio_recent_vs_older'] = (
                recent_3 / older_avg if older_avg > 0
                else (1.0 if recent_3 > 0 else 0.0)
            )

            last_month = float(monthly.iloc[-1])
            prev_avg = float(monthly.iloc[:-1].mean()) if len(monthly) > 1 else 0.0
            f['last_month_vs_previous_avg'] = (
                last_month / prev_avg if prev_avg > 0
                else (1.0 if last_month > 0 else 0.0)
            )
        else:
            f['activity_ratio_recent_vs_older'] = 1.0
            f['last_month_vs_previous_avg'] = 1.0

        return f

    @staticmethod
    def _empty_temporal() -> Dict:
        return {
            'days_since_last_communication': 999,
            'months_since_last_communication': 999,
            'has_communication_last_30d': 0,
            'has_communication_last_60d': 0,
            'has_communication_last_90d': 0,
            'relationship_duration_days': 0,
            'relationship_duration_months': 0,
            'total_active_months': 0,
            'avg_interval_days': 0,
            'std_interval_days': 0,
            'max_interval_days': 0,
            'min_interval_days': 0,
            'days_since_last_vs_avg_interval': 999,
            'is_overdue': 0,
            'activity_ratio_recent_vs_older': 0,
            'last_month_vs_previous_avg': 0,
        }

    # ------------------------------------------------------------------ общее
    def create_base_features(
            self,
            df_clients: pd.DataFrame,
            inn_col: str = 'INN',
            date_col: str = 'upper_bound',
    ) -> pd.DataFrame:
        df = df_clients.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # нормализуем ИНН в клиентском df
        df['__inn_norm'] = df[inn_col].apply(self._normalize_inn)

        # статистика совпадений (для отладки)
        client_inns = set(df['__inn_norm'].dropna().unique())
        stats_inns = set(self._by_inn.keys())
        common = client_inns & stats_inns
        print(f"[engineer] клиентов: {len(df)}, "
              f"ИНН клиентов: {len(client_inns)}, "
              f"ИНН в статистике: {len(stats_inns)}, "
              f"пересечение: {len(common)}")

        rows = []
        for _, row in df.iterrows():
            inn = row['__inn_norm']
            ub = row[date_col]

            if pd.isna(ub) or inn is None:
                feats = self._empty_features(ub)
            else:
                client = self._by_inn.get(inn)
                if client is None or len(client) == 0:
                    feats = self._empty_features(ub)
                else:
                    feats = {}
                    for months, prefix in self.PERIODS:
                        feats.update(self._period_features(client, ub, months, prefix))
                    feats.update(self._derivative_features(client, ub))
                    feats.update(self._trend_features(client, ub))
                    feats.update(self._activity_features(client, ub))
                    feats.update(self._diversity_features(client, ub))
                    feats.update(self._get_seasonal_features(client, ub))
                    feats.update(self._temporal_features(client, ub))

            feats[inn_col] = row[inn_col]
            feats[date_col] = ub
            rows.append(feats)

        out = pd.DataFrame(rows)

        all_cols = self._all_feature_names()
        out = out.reindex(columns=sorted(all_cols | {inn_col, date_col}))
        out = out.fillna(0)

        # убираем возможные одноимённые столбцы из df, чтобы merge не дал _x/_y
        common_cols = [c for c in df.columns if c in out.columns
                       and c not in (inn_col, date_col)]
        df = df.drop(columns=common_cols + ['__inn_norm'], errors='ignore')

        return df.merge(out, on=[inn_col, date_col], how='left')

    def _empty_features(self, upper_bound) -> Dict:
        """Все нули по полной схеме."""
        return {name: 0 for name in self._all_feature_names()}


# =============================================================================
# Удобные обёртки
# =============================================================================

def process_mail_tags(
    mapping_file: str = 'data_cur/mapping/merged.csv',
    mail_file: str = 'data_cur/mail_tagged_plus_llm.csv',
    calls_file: str = 'data_cur/calls_plus_llm.csv',
    statistics_file: str = 'tag_statistics_by_inn_monthly.csv',
    intermediate_file: str = 'mail_calls_with_inn.csv',
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    proc = MailTagProcessor(
        mapping_file=mapping_file,
        mail_file=mail_file,
        calls_file=calls_file,
        **kwargs,
    )
    stats = proc.run_pipeline(
        save=True,
        statistics_file=statistics_file,
        intermediate_file=intermediate_file,
    )
    return stats, proc.get_matched_data()


def create_attrition_features(
    df_statistics: pd.DataFrame,
    df_clients: pd.DataFrame,
    inn_col: str = 'INN',
    date_col: str = 'upper_bound',
    include_all: bool = True,
) -> pd.DataFrame:
    eng = MailTagFeatureEngineer(df_statistics)
    return eng.create_base_features(df_clients, inn_col=inn_col, date_col=date_col)


# =============================================================================
# Пример запуска
# =============================================================================

if __name__ == '__main__':
    statistics, matched = process_mail_tags()

    df_clients = pd.read_csv('data/v23/train_2025.csv')
    df_features = create_attrition_features(
        df_statistics=statistics,
        df_clients=df_clients,
        inn_col='INN',
        date_col='upper_bound',
    )
    print(df_features.head())
    df_features.to_csv('data/v24/train2025.csv', index=False)