import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List
from transformers import pipeline
from typing import List, Union
import sys
sys.path.append('.')
from collabllm.models import get_meta_info_from_model_name, is_unsloth_model_auto
from collabllm.utils.extract_json_reliable import extract_json

# Dictionary to cache loaded Hugging Face models and tokenizers
loaded_hf_models = {}

def generation_pipeline_hf(message: Union[str, List[dict]],
                           model: torch.nn.Module, 
                           tokenizer: AutoTokenizer,
                           stop_sequence=[], 
                           device: str="auto", 
                           json_object=False,
                           **kwargs):

        torch.cuda.empty_cache()
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.padding_side = 'left'   
        with torch.no_grad():
            generator = pipeline("text-generation", 
                                 model=model, tokenizer=tokenizer,
                                 model_kwargs={"torch_dtype": "auto"},
                                 device_map=device)
            terminators = [generator.tokenizer.eos_token_id]
            if len(stop_sequence) > 0:
                terminators = terminators + [generator.tokenizer.convert_tokens_to_ids(s) for s in stop_sequence]
                
            outputs = generator(
                message,
                eos_token_id=terminators,
                **kwargs
            )
            
        if isinstance(message, str):
            result = outputs[0]["generated_text"][len(message):]
        else:
            result = outputs[0]['generated_text'][-1]['content']
        if result is None or not isinstance(result, str):
            result = ' '

        del outputs
        torch.cuda.empty_cache()

        if json_object:
            import pdb; pdb.set_trace()
            return extract_json(result)
        else:
            return result


def generate_text_hf(message: Union[str, List[dict]], 
                     model: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                     max_new_tokens: int = 2048, 
                     temperature: float = 0.5, 
                     json_object: bool = False,
                     stop_sequence: list = [], 
                     cached_model=False, 
                     **kwargs) -> str:
    """
    Generate text completion using a specified Hugging Face model.

    Args:
        message (str): The input text message for completion.
        model (str): The Hugging Face model to use. Default is "huggingface/codellama/CodeLlama-7b-hf".
        max_new_tokens (int): The maximum number of tokens to generate. Default is 2000.
        temperature (float): Sampling temperature for generation. Default is 0.5.
        json_object (bool): Whether to format the message for JSON output. Default is False.
        max_retry (int): Maximum number of retries in case of an error. Default is 1.
        sleep_time (int): Sleep time between retries in seconds. Default is 0.
        stop_sequence (list): List of stop sequences to halt the generation.
        **kwargs: Additional keyword arguments for the `generate` function.

    Returns:
        str: The generated text completion.
    """
    
    if json_object and not 'json' in message:
        message = "You are a helpful assistant designed to output in JSON format." + message
    
    model_name = model.split("/", 1)[1]
    
    # Load the model and tokenizer if not already loaded
    if model_name in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model_name]
    else:
        if is_unsloth_model_auto(model):
            from unsloth import FastLanguageModel
            hf_model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model,
                max_seq_length=max_new_tokens,
                load_in_4bit=True,
                dtype=None
            )
            FastLanguageModel.for_inference(hf_model) 
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype='auto')
            tokenizer = AutoTokenizer.from_pretrained(model)

    if cached_model:
        loaded_hf_models[model_name] = (hf_model, tokenizer)
    stop_sequence = get_meta_info_from_model_name(model_name)['stop_sequence']

    text = generation_pipeline_hf(message, 
                                  hf_model, tokenizer, 
                                  stop_sequence=stop_sequence, 
                                  temperature=temperature, 
                                  max_new_tokens=max_new_tokens, 
                                  **kwargs)

    if json_object:
        return extract_json(text)
    else:
        return text


# if __name__ == "__main__":
#     message = "Hello, how are you today?"
#     model = "meta-llama/Meta-Llama-3-8B-Instruct" # "meta-llama/Meta-Llama-3-70B-Instruct" 
#     # "meta-llama/Meta-Llama-3-8B-Instruct" # "unsloth/llama-3-70b-bnb-4bit"
#     # completion = generate_text_hf(message, model=model, max_new_tokens=16)
#     # print(completion)
#     completion = generate_text_hf("You are a helpful assistant. Given a conversation between a user and a chat assistant, summarize the chat assistant's final answer to the user's target question in 1-3 sentences. Follow the format and style provided in the examples below:\n**user**: I need help finding the equation of a plane that passes through two points and is perpendicular to another plane. \n**assistant**: Sure, I can help with that! To find the equation of a plane that passes through two points and is perpendicular to another plane, we need to follow these steps:1. **Understand and List Given Information:** - The coordinates of the two points.- The equation of the plane to which the new plane is perpendicular. Could you please provide: 1. The coordinates of the two points (let's call them \( A(x_1, y_1, z_1) \) and \( B(x_2, y_2, z_2) \)). 2. The equation of the plane to which the new plane should be perpendicular (in the form \( Ax + By + Cz + D = 0 \)). Once I have this information, I'll guide you through the calculation step-by-step! **assistant**: Great, we have all the information we need! The equation of the desired plane is: \[ x + y - z + 1 = 0 \]\nThe answerized answer should only be based on the given information. Provide the summary directly without additional comments.\nSummarized Answer: ", model="meta-llama/Meta-Llama-3-8B-Instruct" , max_new_tokens=1024, max_length=100, truncation=True)