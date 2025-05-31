from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Encode input and generate output
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    prompt = input("Enter your topic or sentence: ")
    generated_text = generate_text(prompt)
    print("\nGenerated Text:\n", generated_text)
