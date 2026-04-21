from transformers import AutoModelForMaskedLM, AutoTokenizer

#################################
## CONSTANTS
#################################

AVAILABLE_MODELS = [
        "albert"
        ]   
            
MODEL_CONFIGS = {
        "albert": {
            "name": "albert/albert-base-v2",
            "library": "transformers",
        }
}


#################################
## Class definitions
#################################


#################################
## Private Functions
#################################

def _get_transformers_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config["name"])
    model = AutoModelForMaskedLM.from_pretrained(config["name"])
    return {
        "tokenizer": tokenizer,
        "model": model,
        "model_name": config["name"],
        "type": "causal-lm",
    }

#################################
## Public Functions
#################################

def get_model(name):
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{name}' not found. Available models: {AVAILABLE_MODELS}")
    
    config = MODEL_CONFIGS[name]
    
    if config["library"] == "transformers":
        return _get_transformers_model(config)
        

#################################
## Test function
#################################

if __name__ == "__main__":
    get_model("albert")
 
