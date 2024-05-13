import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
base_path = './final_model'  #模型下载的文件夹

#os.system(f'git clone 模型地址.git {base_path}') #这里的模型地址就是你模型的下载地址，简单说就是前面创建模型时候git clone的同样代码
os.system(f'git@code.openxlab.org.cn:zhuyamei/final_model.git {base_path}')

os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()
