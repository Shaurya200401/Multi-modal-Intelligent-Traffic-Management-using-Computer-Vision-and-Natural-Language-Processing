import pandas as pd
from nlp.relevance_filter import is_traffic_relevant

class TrafficTextFeed:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["TWEET"])
        self.texts = df["TWEET"].tolist()
        self.index = 0

    def get_next(self):
        while True:
            if self.index >= len(self.texts):
                self.index = 0

            text = self.texts[self.index]
            self.index += 1

            if is_traffic_relevant(text):
                return text
    def is_area_relevant(text, area_name):
        return area_name.lower() in text.lower()
