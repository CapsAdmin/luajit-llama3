Llama3 inference with luajit using the Q4_0 model variant.

99% of the time is being spent in Tensor.MatrixVectorMultiply. There is a pthreads and a cuda variant, though both are not optimal. The cuda version spends most of its time uploading tensors, and the pthreads version is running in multiple lua states spread accross cores.

The cuda version is however significantly faster.

It would be cool to make the pure luajit version faster, but I'm not really sure how.

To try the cuda version, change use_pthreads() in llama.lua to use_cuda(). It depends on libnvrtc for compiling kernels and libcuda for everything else.

```
luajit llama.lua cuda "Meta-Llama-3-8B-Instruct-Q4_0.gguf" "what is luajit?"
<|start_header_id|>user<|end_header_id|>
what is luajit?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

LuaJIT is a just-in-time (JIT) compiler for the Lua programming language. It is designed to provide high-performance execution of Lua code, making it suitable for use in applications where speed and efficiency are critical.

LuaJIT is a se^C
```

I mostly used https://github.com/mukel/llama3.java as source reference. You can find the instructions on how to download "Meta-Llama-3-8B-Instruct-Q4_0.gguf" in there.

