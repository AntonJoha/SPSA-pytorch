from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

#################################
## CONSTANTS
#################################


AVAILABLE_MODELS = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "gpt2",
    "qlora",
    "qlora-gpt2",
]

#################################
## Class definitions
#################################


#################################
## Functions
#################################




def _masked_lm(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "model_name": name,
        "type": "masked-lm",
    }


def _causal_lm(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "model_name": name,
        "type": "causal-lm",
    }


def _qlora(base_model_name="gpt2", lora_r=16, lora_alpha=32, lora_dropout=0.05):
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as e:
        raise ImportError(
            "QLoRA support requires `peft` to be installed."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
    except Exception as e:
        raise RuntimeError(
            "QLoRA model loading failed. Install bitsandbytes-compatible dependencies "
            "and ensure your environment supports 4-bit quantization."
        ) from e

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "model_name": base_model_name,
        "type": "qlora-causal-lm",
    }


def burt_base_uncased():
    return _masked_lm("bert-base-uncased")


def get_model(name, **kwargs):
    if name in ("bert-base-uncased", "distilbert-base-uncased"):
        return _masked_lm(name)
    if name == "gpt2":
        return _causal_lm(name)
    if name in ("qlora", "qlora-gpt2"):
        base_model_name = kwargs.pop("base_model_name", "gpt2")
        return _qlora(base_model_name=base_model_name, **kwargs)
    raise ValueError(
        f"Unknown model '{name}'. Available models: {', '.join(AVAILABLE_MODELS)}"
    )



#################################
## Test function
#################################

if __name__ == "__main__":
    print("Test?")
