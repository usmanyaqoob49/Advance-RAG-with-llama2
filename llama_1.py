#configuration of llama2 
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch import cuda, bfloat16

def model_tokenizer():
    from huggingface_hub import notebook_login
    notebook_login('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')

    model_id= "meta-llama/Llama-2-7b-chat-hf"

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
    )

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library


    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=bfloat16
    # )


    model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    # quantization_config=bnb_config,
    device_map='auto',
    )



    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer