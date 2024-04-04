
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from transformers import BertTokenizer, BertModel
from transformers import RemBertModel, RemBertConfig, RemBertTokenizer

#unmasker = pipeline("fill-mask", model="xlm-roberta-large")
#unmasker = pipeline("fill-mask", model="xlm-roberta-base")


#model_name = "Davlan/afriberta_large"

# Download and cache model and tokenizer
#model = AutoModel.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# The model and tokenizer are now cached in the Hugging Face cache directory

#tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-large")
#model = AutoModelForMaskedLM.from_pretrained("Davlan/afro-xlmr-large")

#model = XLMRobertaModel.from_pretrained("bonadossou/afrolm_active_learning")
#tokenizer = XLMRobertaTokenizer.from_pretrained("bonadossou/afrolm_active_learning")
#tokenizer.model_max_length = 256


#mBERT

#model = BertModel.from_pretrained("bert-base-multilingual-cased")
#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#RemBERT

#configuration = RemBertConfig()
#model = RemBertModel(configuration)

#configuration = model.config

model = RemBertModel.from_pretrained("google/rembert")
tokenizer = RemBertTokenizer.from_pretrained('google/rembert')


"""

#XLMR
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

"""
