import tiktoken 

#use gpt tokenizer

encoding =tiktoken.encoding_for_model("gpt-4o-mini")
sentences={
    "English":"Artificial intelligence is dumb",
    "code":"def add(a,b): return a+b"
}
token_count={}
for lang,text in sentences.items():
    tokens=encoding.encode(text)
    token_count[lang]=len(tokens)
    print(f"\n {lang} Sentence:{text}")
    print("Tokens:",tokens)
    print("Token Count:",len(tokens))