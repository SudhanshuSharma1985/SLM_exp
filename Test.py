from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Sudhanshu1985/slm-125m-base")
tok   = AutoTokenizer.from_pretrained("Sudhanshu1985/slm-125m-base")
print(tok)