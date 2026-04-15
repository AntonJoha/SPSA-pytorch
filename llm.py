from transformers import (
    BertTokenizer,
    BertForMaskedLM,
)

#################################
## CONSTANTS
#################################


AVAILABLE_MODELS = [
    "bert-base-uncased",
]

#################################
## Class definitions
#################################


#################################
## Functions
#################################




def burt_base_uncased():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    return {
        "tokenizer": tokenizer,
        "model": model
    }

def get_model(name):
    if name == "bert-base-uncased":
        return burt_base_uncased()
    return None



#################################
## Test function
#################################

if __name__ == "__main__":
    print("Test?")