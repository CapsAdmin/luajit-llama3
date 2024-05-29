Llama3 inference with luajit using the Q4_0 model variant. It's very slow at the moment..

```
luajit llama.lua /home/caps/projects/llama3.java/Meta-Llama-3-8B-Instruct-Q4_0.gguf
reading gguf metadata took 0.095998 seconds
reading gguf tensors took 1.810503 seconds
<|start_header_id|>user<|end_header_id|>
hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'm happy to chat with you^C
```

I mostly used https://github.com/mukel/llama3.java as source reference.
