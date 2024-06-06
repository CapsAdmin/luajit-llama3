Llama3 inference with luajit using the Q4_0 model variant. 

It can use cuda, pthreads or plain luajit to do inference, though the luajit variant is painfully slow. Most of the time is being spent in Tensor.MatrixVectorMultiply

The cuda version uses a kernel to do the multiplication while the pthreads version just spreads the calculation accross multiple lua states in threads. Otherwise all calculations are done in luajit.

It would be cool to make the pure luajit version faster, but I'm not really sure how. Using simd can speed it up quite a bit, but this is not available directly in LuaJIT ([though it's on the roadmap](https://github.com/LuaJIT/LuaJIT/pull/116)) so the only option is to compile specialized C code to load with ffi.


```
luajit llama.lua cuda "Meta-Llama-3-8B-Instruct-Q4_0.gguf" "write a luajit haiku"
cuda driver version: 12.04
using device: NVIDIA GeForce RTX 4090
reading gguf metadata took 0.10612 seconds
reading gguf tensors took 2.02863 seconds
uploading tensors to gpu took 0.50892 seconds
4.21gb tensors allocated on GPU
3.35 / 23.64 gb vram in use
4.33gb tensors allocated on CPU




<|start_header_id|>user<|end_header_id|>
write a luajit haiku<|eot_id|><|start_header_id|>assistant<|end_header_id|>
A haiku in Lua!

Fuzzy math whispers
Glowing pixels unfold
Code's gentle hum<|eot_id|>



3.36 / 23.64 gb vram in use
token count: 38
elapsed: 1.86s
20.40 tokens/s
```

I mostly used https://github.com/mukel/llama3.java as source reference. You can find the instructions on how to download "Meta-Llama-3-8B-Instruct-Q4_0.gguf" in there.

