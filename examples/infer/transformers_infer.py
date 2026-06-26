from transformers import AutoModelForCausalLM, AutoTokenizer

# loading model
model = AutoModelForCausalLM.from_pretrained(
    model_dir, # Model checkpoints 
    torch_dtype=torch.float16, 
    trust_remote_code=True,
    device_map="auto"
)


#loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True
)

# construct message
user_content =  """你是一个影视剧语言翻译专家，擅长根据影视剧对白并结合语境把中文翻译成其他语言给定影视剧的一段上下文对白，结合当前影视剧的语境,将指定的对白翻译成英语
## 格式要求
- 输入输出格式必须完全一致：JSON格式，键为字幕编号，值为翻译内容
- 确保翻译后的条目数量与原始字幕完全一致
- 对专有名词、术语严格按照术语表执行
- 给定上下文，翻译当前文本，不要遗漏任何字幕

## 输出格式
严格按照以下JSON格式输出，不得有任何额外内容:
'''json
{
  "1": "翻译内容1",
  "2": "翻译内容2",
  ...
}
'''
[上文]
启禀太后
皇上病重
但不见任何人
臣妾们担心的紧
不见任何人?

[当前文本] 
'''json{
  "1": "母后",
  "2": "既然皇上不见人",
  "3": "要不咱们还是走吧?"
}'''

[下文] 
你给我回来
皇上传皇后觐见
"""

messages = [{"role": "user", "content": user_content}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=1024)

input_len = len(inputs.input_ids[0])	
generated_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

print(f"translated_text: {generated_text}")