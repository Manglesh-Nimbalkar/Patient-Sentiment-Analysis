import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

token = "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"

tokenizer = AutoTokenizer.from_pretrained("Mohammed-Altaf/Medical-ChatBot", config={"token": token})
model = AutoModelForCausalLM.from_pretrained("Mohammed-Altaf/Medical-ChatBot", config={"token": token})

device = "cuda" if torch.cuda.is_available() else "cpu"

prompt_input = (
    "The conversation between human and AI assistant.\n"
    "[|Human|] {input}\n"
    "[|AI|]"
)
sentence = prompt_input.format_map({'input': "what is parkinson's disease?"})
inputs = tokenizer(sentence, return_tensors="pt").to(device)

with torch.no_grad():
    beam_output = model.generate(**inputs,
                                min_new_tokens=1, 
                                max_length=512,
                                num_beams=3,
                                repetition_penalty=1.2,
                                early_stopping=True,
                                eos_token_id=tokenizer.eos_token_id 
                                )
    print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
