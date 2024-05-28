Llama3 inference with luajit using the Q4_0 model variant. It's very slow at the moment..

```
[caps@nixos:~/projects/luajit-llama3]$ luajit llama3.lua /home/caps/projects/llama3.java/Meta-Llama-3-8B-Instruct-Q4_0.gguf
reading gguf metadata took 0.098627 seconds
reading gguf tensors took 1.765626 seconds
<|start_header_id|>
user
<|end_header_id|>
Ċ
hello
<|eot_id|>
<|start_header_id|>
assistant
<|end_header_id|>
ĊĊ
Hello
!
ĠIt
's
Ġnice
Ġto
^C^C
```

I mostly used https://github.com/mukel/llama3.java as source reference.
