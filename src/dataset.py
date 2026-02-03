import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import DistilBertTokenizer

class FinancialDataset(Dataset):

    def __init__(self, news_path, prices_path, tokenizer_name= 'distilbert-base-uncased', max_len = 128):
        #args : csv_path - path to csv file, json_path - path to json_file, tokenizer_name - the hugging face model used, max_len - max token length

        self.news_df = pd.read_csv(news_path)
        self.prices_df = pd.read_csv(prices_path)

        #merging the data with data as a PK
        self.data = pd.merge(self.news_df, self.prices_df, on='Date')

        #sorting data
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)

        #clean text data
        self.news_cols = [c for c in self.data.columns if c.startswith('Top')]

        #fill missing data
        self.data[self.news_cols] = self.data[self.news_cols].fillna('')

        #normalize numbers
        self.numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        for col in self.numerical_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.data = self.data.dropna(subset=self.numerical_cols)

        self.num_means = self.data[self.numerical_cols].mean()
        self.num_stds = self.data[self.numerical_cols].std()

        #initialize Tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        # to know how many samples exist
        return len(self.data)

    def __getitem__(self, idx):
        #returns one sample of data

        ##Processing text ##
        row = self.data.iloc[idx]
        headlines = " ".join([str(row[col]) for col in self.news_cols])

        #cleaning data
        headlines = headlines.replace('b"','').replace('b\'','')

        encoding = self.tokenizer(
            headlines,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        ## processing numerical features ##

        raw_values = row[self.numerical_cols].values.astype('float32')
        means = self.num_means.values.astype('float32')
        stds = self.num_stds.values.astype('float32')

        normalized_values = (raw_values - means) / stds
        price_tensor = torch.tensor(normalized_values)
        ## labelling ##
        #already done in data set
        label = int(row['Label'])

        return{
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'price_features' : price_tensor,
            'label' : torch.tensor(label, dtype = torch.long)
        }


