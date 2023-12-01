from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

model_name = "castorini/afriberta_large"

# Download and cache the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)