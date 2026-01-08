import yfinance as yf
import pandas as pd


class StockDataLoader:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self) -> pd.DataFrame:
        df = yf.download(
            self.symbol,
            start=self.start_date,
            end=self.end_date
        )
        return df

    def save_to_csv(self, df: pd.DataFrame, path: str):
        df.to_csv(path)
