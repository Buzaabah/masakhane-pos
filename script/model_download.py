
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from transformers import XLMRobertaModel, XLMRobertaTokenizer

pipe = pipeline("fill-mask", model="bonadossou/afrolm_active_learning")

#model_name = "Davlan/afriberta_large"

# Download and cache model and tokenizer
#model = AutoModel.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# The model and tokenizer are now cached in the Hugging Face cache directory

#tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-large")
#model = AutoModelForMaskedLM.from_pretrained("Davlan/afro-xlmr-large")

model = XLMRobertaModel.from_pretrained("bonadossou/afrolm_active_learning")
tokenizer = XLMRobertaTokenizer.from_pretrained("bonadossou/afrolm_active_learning")
tokenizer.model_max_length = 256