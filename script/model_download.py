
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

pipe = pipeline("fill-mask", model="Davlan/afro-xlmr-base")

#model_name = "Davlan/afriberta_large"

# Download and cache model and tokenizer
#model = AutoModel.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# The model and tokenizer are now cached in the Hugging Face cache directory

tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-base")
model = AutoModelForMaskedLM.from_pretrained("Davlan/afro-xlmr-base")