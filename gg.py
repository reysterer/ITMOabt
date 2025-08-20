# gg.py
# Универсальный EDA для transaction_fraud_data.parquet
# Режимы:
#   python gg.py                -> базовый EDA (без конвертации)
#   python gg.py --usd          -> EDA с конвертацией в USD (использует historical_currency_exchange.parquet)


import argparse
import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)


def load_transactions(path: str) -> pd.DataFrame:
    print(f"Загружаем транзакции из: {path}")
    df = pd.read_parquet(path)
    print("Размер датасета:", df.shape)
    print("\nПервые строки:")
    print(df.head())
    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("\nТипы данных:")
    print(df.dtypes)

    print("\nПропуски по столбцам:")
    print(df.isna().sum())

    print("\nОсновная статистика по числовым признакам:")
    print(df.describe())

    # Баланс классов (если есть is_fraud)
    if "is_fraud" in df.columns:
        fraud_counts = df["is_fraud"].value_counts(normalize=True) * 100
        print("\nДоля мошеннических операций (%):")
        print(fraud_counts)

    # Средний чек по классам
    if "is_fraud" in df.columns and "amount" in df.columns:
        avg_amount = df.groupby("is_fraud")["amount"].mean().sort_index()
        print("\nСредний чек по классам (amount):")
        print(avg_amount)

    # ТОП-10 городов по количеству транзакций
    if "city" in df.columns:
        top_cities = df["city"].value_counts().head(10)
        print("\nТОП-10 городов по количеству транзакций:")
        print(top_cities)

    # ТОП-10 городов по среднему чеку (в исходной валюте)
    if "city" in df.columns and "amount" in df.columns:
        city_avg = df.groupby("city")["amount"].mean().sort_values(ascending=False).head(10)
        print("\nТОП-10 городов по среднему чеку (amount):")
        print(city_avg)

    # Средний чек по категориям продавцов
    if "vendor_category" in df.columns and "amount" in df.columns:
        vc_avg = df.groupby("vendor_category")["amount"].mean().sort_values(ascending=False).head(10)
        print("\nТОП-10 vendor_category по среднему чеку (amount):")
        print(vc_avg)

    # Средний чек по типам продавцов
    if "vendor_type" in df.columns and "amount" in df.columns:
        vt_avg = df.groupby("vendor_type")["amount"].mean().sort_values(ascending=False).head(10)
        print("\nТОП-10 vendor_type по среднему чеку (amount):")
        print(vt_avg)


def load_fx_wide(fx_path: str) -> pd.DataFrame:
    print(f"\nЗагружаем курсы валют из: {fx_path}")
    fx = pd.read_parquet(fx_path)

    if "date" not in fx.columns:
        raise ValueError("В файле курсов нет столбца 'date'.")
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce").dt.date

    # длинный формат -> pivot (date, currency, rate)
    if {"currency", "rate"}.issubset(set(fx.columns)):
        fx_wide = fx.pivot_table(index="date", columns="currency", values="rate", aggfunc="last").sort_index()
    else:
        # уже широкий: date + столбцы-валюты
        fx_wide = fx.set_index("date").sort_index()

    # тянем курсы вперёд для пропусков
    fx_wide = fx_wide.ffill()
    print("Курсы загружены. Формат широкий. Форм shape:", fx_wide.shape)
    return fx_wide


def convert_to_usd(df: pd.DataFrame, fx_wide: pd.DataFrame) -> pd.DataFrame:
    # готовим дату для join
    if "timestamp" not in df.columns:
        raise ValueError("В транзакциях нет столбца 'timestamp' для сопоставления по дате.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date

    merged = df.merge(fx_wide, left_on="date", right_index=True, how="left")

    def to_usd(row):
        amt = row.get("amount")
        cur = row.get("currency")
        if pd.isna(amt) or pd.isna(cur):
            return pd.NA
        if cur == "USD":
            return amt
        rate = row.get(cur)
        if pd.isna(rate):
            return pd.NA
        # Конвенция: rate = сколько единиц валюты за 1 USD
        return amt / rate

    merged["amount_usd"] = merged.apply(to_usd, axis=1)
    missing = merged["amount_usd"].isna().sum()
    print(f"\nНе удалось сконвертировать в USD строк: {missing} из {len(merged)}")
    return merged


def usd_eda(df_usd: pd.DataFrame) -> None:
    # Средний чек USD по классам
    if "is_fraud" in df_usd.columns and "amount_usd" in df_usd.columns:
        avg_by_fraud = (
            df_usd.dropna(subset=["amount_usd"])
                  .groupby("is_fraud")["amount_usd"]
                  .mean()
                  .sort_index()
        )
        print("\nСредний чек (USD) по классам (is_fraud):")
        print(avg_by_fraud)

    # ТОП-10 городов по среднему чеку (USD)
    if "city" in df_usd.columns and "amount_usd" in df_usd.columns:
        city_avg_usd = (
            df_usd.dropna(subset=["amount_usd"])
                  .groupby("city")["amount_usd"]
                  .mean()
                  .sort_values(ascending=False)
                  .head(10)
        )
        print("\nТОП-10 городов по среднему чеку (USD):")
        print(city_avg_usd)

    # ТОП-10 vendor_category по среднему чеку (USD)
    if "vendor_category" in df_usd.columns and "amount_usd" in df_usd.columns:
        vc_avg_usd = (
            df_usd.dropna(subset=["amount_usd"])
                  .groupby("vendor_category")["amount_usd"]
                  .mean()
                  .sort_values(ascending=False)
                  .head(10)
        )
        print("\nТОП-10 vendor_category по среднему чеку (USD):")
        print(vc_avg_usd)

    # ТОП-10 vendor_type по среднему чеку (USD)
    if "vendor_type" in df_usd.columns and "amount_usd" in df_usd.columns:
        vt_avg_usd = (
            df_usd.dropna(subset=["amount_usd"])
                  .groupby("vendor_type")["amount_usd"]
                  .mean()
                  .sort_values(ascending=False)
                  .head(10)
        )
        print("\nТОП-10 vendor_type по среднему чеку (USD):")
        print(vt_avg_usd)


def main():
    parser = argparse.ArgumentParser(description="EDA для transaction_fraud_data.parquet (с опцией пересчёта в USD)")
    parser.add_argument("--usd", action="store_true", help="Выполнить конвертацию в USD и сводки по amount_usd")
    parser.add_argument("--file", default="transaction_fraud_data.parquet", help="Путь к файлу транзакций")
    parser.add_argument("--fx", default="historical_currency_exchange.parquet", help="Путь к файлу курсов")
    args = parser.parse_args()

    # 1) Базовая загрузка и EDA
    df = load_transactions(args.file)
    basic_eda(df)

    # 2) По желанию — конвертация в USD и EDA по amount_usd
    if args.usd:
        fx_path = Path(args.fx)
        if not fx_path.exists():
            raise FileNotFoundError(f"Файл курсов не найден: {fx_path}")
        fx_wide = load_fx_wide(str(fx_path))
        df_usd = convert_to_usd(df, fx_wide)
        usd_eda(df_usd)

    print("\nГотово ✅")


if __name__ == "__main__":
    main()