import torch
import torch.nn as nn
from transformers import DistilBertModel

class MultimodalPredictor(nn.Module):
    def __init__(self, num_numerical_features=5, hidden_size=64, num_classes=2):
       #num_numerical_features - no' of inputs in the price vector
       #hidden size - size of internal layers
       #num_classes = 2 for binary

        super(MultimodalPredictor, self).__init__()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

       #Text processing through BERT model, pretrained with english grammar/vocab
        for param in self.bert.parameters():
            param.requires_grad = False

        #reducing overfitting
        self.text_drop = nn.Dropout(0.3)

        #Numerical processing - feed forward network
        self.text_out_layer = nn.Linear(768, hidden_size)
        self.numerical_layer = nn.Linear(num_numerical_features, hidden_size)
        self.numerical_relu = nn.ReLU()

        #merging by concatenating text and numerical processing
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask, price_features):

        #processing text
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        #summary token
        cls_token = bert_output.last_hidden_state[:, 0, :]

        text_feat = self.text_drop(cls_token)
        text_feat = self.text_out_layer(text_feat)

        #Number processing
        num_feat = self.numerical_layer(price_features)
        num_feat = self.numerical_relu(num_feat)


        #merging information
        combined = torch.cat((text_feat, num_feat), dim=1)

        logits = self.classifier(combined)

        return logits