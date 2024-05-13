import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
base_path = './final_model'  #模型下载的文件夹

#os.system(f'git clone 模型地址.git {base_path}') #这里的模型地址就是你模型的下载地址，简单说就是前面创建模型时候git clone的同样代码
#os.system(f'git@code.openxlab.org.cn:zhuyamei/final_model.git {base_path}')
os.system(f'git clone https://code.openxlab.org.cn/zhuyamei/final_model.git {base_path}')

os.system(f'cd {base_path} && git lfs pull')

model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-1_8B",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
