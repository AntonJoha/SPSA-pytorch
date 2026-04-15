import warnings

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


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _causal_lm(name):
    tokenizer = _ensure_pad_token(AutoTokenizer.from_pretrained(name))
    model = AutoModelForCausalLM.from_pretrained(name)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "model_name": name,
        "type": "causal-lm",
    }


def _qlora(base_model_name="gpt2", lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Load a 4-bit quantized causal LM and wrap it with LoRA adapters.
    """
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as e:
        raise ImportError(
            "QLoRA support requires `peft` to be installed. "
            "Install it with: pip install peft"
        ) from e

    tokenizer = _ensure_pad_token(AutoTokenizer.from_pretrained(base_model_name))

    # QLoRA 4-bit setup: NF4 + double quantization for memory-efficient training.
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


def bert_base_uncased():
    return _masked_lm("bert-base-uncased")


def burt_base_uncased():
    """Backward-compatible alias for previous misspelled public API."""
    warnings.warn(
        "`burt_base_uncased` is deprecated; use `bert_base_uncased` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bert_base_uncased()


def get_model(name, **kwargs):
    if name in ("bert-base-uncased", "distilbert-base-uncased"):
        return _masked_lm(name)
    if name == "gpt2":
        return _causal_lm(name)
    if name in ("qlora", "qlora-gpt2"):
        base_model_name = kwargs.get("base_model_name", "gpt2")
        allowed_qlora_kwargs = {"lora_r", "lora_alpha", "lora_dropout"}
        unknown_kwargs = set(kwargs) - allowed_qlora_kwargs - {"base_model_name"}
        if unknown_kwargs:
            raise TypeError(
                f"Unknown QLoRA kwargs: {', '.join(sorted(unknown_kwargs))}. "
                "Allowed kwargs: base_model_name, lora_r, lora_alpha, lora_dropout."
            )
        qlora_kwargs = {k: v for k, v in kwargs.items() if k in allowed_qlora_kwargs}
        return _qlora(base_model_name=base_model_name, **qlora_kwargs)
    raise ValueError(
        f"Unknown model '{name}'. Available models: {', '.join(AVAILABLE_MODELS)}. "
        "Note: QLoRA entries accept optional kwargs such as "
        "base_model_name, lora_r, lora_alpha, and lora_dropout."
    )



#################################
## Test function
#################################

if __name__ == "__main__":
    print("Test?")
