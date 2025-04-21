from transformers import AutoTokenizer

def init_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

tokenizer = init_tokenizer("/home/lipeng/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

def test_tokenizer():
    # 测试编码功能
    text = "这是一个测试文本   aaa bbb"
    tokens = tokenizer(text, return_tensors="np")
    print(f"{text}==>{tokens.input_ids.shape} {tokens}")
    
    # 测试解码功能
    for i in range(tokens.input_ids.shape[0]):
        for j in range(tokens.input_ids.shape[1]):
            decoded_text = tokenizer.decode(tokens.input_ids[i][j])
            print(f"{i,j}->{decoded_text}")
    
    # 验证特殊tokens
    print(f"PAD token:{tokenizer.pad_token_id}=  {tokenizer.pad_token}")
    print(f"EOS token:{tokenizer.eos_token_id}=  {tokenizer.eos_token}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # 测试批处理
    batch_texts = ["测试文本一", "另一个测试文本", "第三个测试文本"]
    batch_tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='np')
    print(f"批处理tokens shape: {batch_tokens.input_ids.shape}")
    
    # 测试最大长度限制
    long_text = "这是一个" * 100
    tokens_truncated = tokenizer(long_text, max_length=20, truncation=True, return_tensors="np")
    print(f"截断后的tokens长度: {tokens_truncated.input_ids.shape[1]}")
    
    return True

if __name__ == "__main__":
    print()
    test_result = test_tokenizer()

    print(f"Tokenizer测试完成: {'成功' if test_result else '失败'}")