from transformers import AutoModel, AutoTokenizer

model_name = "Davlan/afriberta_large"

# Download and cache model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# The model and tokenizer are now cached in the Hugging Face cache directory
