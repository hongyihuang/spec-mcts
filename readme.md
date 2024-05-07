Dependencies
1. Triton for custom kernels
2. transformer-engine for fp8 experimental capabilities (not integrated)
3. PyTorch 2.2 & CUDA 12.1

To Run
1. Clone CodeLlama-7b-Instruct-hf and CodeLlama-70b-Instruct-hf into a directory above or directly in this repo. 
2. Use fp16_to_int4.py to convert into a singular int4 quantized model. 
3. Run load_q40.py
