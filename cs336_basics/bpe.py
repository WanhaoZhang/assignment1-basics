import regex as re
from collections import Counter
from typing import List, Dict, Tuple

# GPT-2 的预分词正则表达式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练 Byte-Pair Encoding (BPE) 分词器。
    """
    print(f"开始训练 BPE，目标词表大小: {vocab_size}")

    # ==========================================
    # 0. 读取输入文件
    # ==========================================
    with open(input_path, 'r', encoding='utf-8') as f:
        input_text = f.read()

    # ==========================================
    # 1. 预处理 (Pre-tokenization)
    # ==========================================
    # 统计单词频率: word_counts = { (108, 111, 119): 5, ... }
    word_counts = _pretokenize_and_count(input_text, special_tokens)
    
    # ==========================================
    # 2. 初始化词表
    # ==========================================
    # 初始词表包含 256 个字节 (0-255)
    vocab = {idx: bytes([idx]) for idx in range(256)}
    
    # 计算需要合并的次数
    # 目标大小 - 初始256 - 特殊token数量
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []

    # ==========================================
    # 3. BPE 训练循环
    # ==========================================
    next_token_id = 256
    
    for i in range(num_merges):
        # 3.1 统计当前所有相邻字节对的频率
        pair_counts = _get_pair_stats(word_counts)
        
        if not pair_counts:
            print("没有更多可以合并的字节对了，提前停止。")
            break

        # 3.2 找到频率最高的 Pair
        # 频率相同时，按字典序选大的 (lexicographically greater pair)
        # 按字节值比较，而不是token ID
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))
        
        # 获取该 pair 当前对应的字节内容（用于记录 merges）
        token_bytes_a = vocab[best_pair[0]]
        token_bytes_b = vocab[best_pair[1]]
        
        # 3.3 记录合并规则
        merges.append((token_bytes_a, token_bytes_b))
        vocab[next_token_id] = token_bytes_a + token_bytes_b
        
        print(f"Merge {i+1}/{num_merges}: {best_pair} (freq={pair_counts[best_pair]}) -> {next_token_id} ({vocab[next_token_id]})")
        
        # 3.4 更新 word_counts
        # 将所有出现 best_pair 的地方替换为 next_token_id
        word_counts = _merge_vocab(word_counts, best_pair, next_token_id)
        
        next_token_id += 1

    # ==========================================
    # 4. 添加 Special Tokens
    # ==========================================
    for st in special_tokens:
        vocab[next_token_id] = st.encode("utf-8")
        next_token_id += 1
        
    return vocab, merges


def _pretokenize_and_count(text: str, special_tokens: List[str]) -> Counter[Tuple[int, ...]]:
    """
    执行 regex 切分，转换为 bytes tuple，并统计频率。
    """
    counts = Counter()
    
    # 1. 先用 special tokens 切分文本
    if special_tokens:
        # 使用正则转义，防止特殊字符干扰，例如 '|'
        pattern = "|".join(re.escape(st) for st in special_tokens)
        segments = re.split(pattern, text)
    else:
        segments = [text]

    # 2. 对每一段非特殊文本进行 GPT-2 正则切分
    for segment in segments:
        if not segment:
            continue
        # 使用 findall 找到所有匹配的单词
        words = re.findall(GPT2_SPLIT_PATTERN, segment)
        for word in words:
            # 将字符串转为 UTF-8 字节的整数元组
            # 例如 "hi" -> b'hi' -> (104, 105)
            word_bytes = tuple(word.encode("utf-8"))
            counts[word_bytes] += 1
            
    return counts


def _get_pair_stats(word_counts: Counter[Tuple[int, ...]]) -> Dict[Tuple[int, int], int]:
    """
    统计当前所有单词中相邻 token 对的出现频率。
    """
    pair_counts = Counter()
    
    for word, freq in word_counts.items():
        # word 是一个整数元组，例如 (104, 101, 108, ...)
        # 如果长度小于 2，就没有 pair
        if len(word) < 2:
            continue
            
        # 遍历单词中的每一个相邻对
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq
            
    return pair_counts


def _merge_vocab(
    word_counts: Counter[Tuple[int, ...]], 
    pair_to_merge: Tuple[int, int], 
    new_token_id: int
) -> Counter[Tuple[int, ...]]:
    """
    将 word_counts 中所有的 pair_to_merge 替换为 new_token_id。
    """
    new_counts = Counter()
    p0, p1 = pair_to_merge
    
    for word, freq in word_counts.items():
        # 如果这个单词太短，或者根本不包含我们要合并的第一个 token，直接跳过处理
        if len(word) < 2 or p0 not in word:
            new_counts[word] += freq
            continue
        
        # 重构单词：执行合并
        new_word = []
        i = 0
        while i < len(word):
            # 检查是否匹配我们要合并的 pair (p0, p1)
            if i < len(word) - 1 and word[i] == p0 and word[i+1] == p1:
                new_word.append(new_token_id)
                i += 2  # 跳过两个位置
            else:
                new_word.append(word[i])
                i += 1
        
        new_counts[tuple(new_word)] += freq
    
    return new_counts


# ==========================================
# 测试代码 (根据 PDF 第 7 页的例子)
# ==========================================
if __name__ == "__main__":
    # 文档中的示例文本
    # 注意：为了复现文档结果，这里简单地用空格连接单词，模拟文档中的词频
    # low: 5, lower: 2, newest: 6, widest: 3
    sample_text = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3
    
    print("=== 测试数据 ===")
    print(sample_text[:50] + "...")
    print("==============")

    # 训练 BPE
    # 目标词表设为 260，这样只会发生几次合并，方便观察
    # 初始 256 + 1个特殊token = 257。我们给它空间做 3 次合并。
    vocab, merges = train_bpe(
        input_text=sample_text, 
        vocab_size=256 + 1 + 3,  # 256 base + 1 special + 3 merges
        special_tokens=["<|endoftext|>"]
    )

    print("\n=== 最终 Merges ===")
    for i, (p1, p2) in enumerate(merges):
        print(f"Merge {i+1}: {p1} + {p2}")
        
    print("\n=== 验证结果 ===")
    # 期望结果应类似于文档:
    # 1. 's' + 't' -> 'st' (频率 9)
    # 2. 'e' + 'st' -> 'est' (频率 9)
    # 3. 'l' + 'o' 或者是 'o' + 'w' (频率 7)
    
    expected_m1 = (b's', b't')
    expected_m2 = (b'e', b'st') # 注意：这里的 b'st' 是上一步合并后的字节内容
    
    if len(merges) >= 1 and merges[0] == expected_m1:
        print("✅ 第 1步合并正确: ('s', 't')")
    else:
        print(f"❌ 第 1步合并错误，期望 ('s', 't')，实际 {merges[0] if merges else 'None'}")
        
    if len(merges) >= 2 and merges[1] == expected_m2:
        print("✅ 第 2步合并正确: ('e', 'st')")
    else:
        print(f"❌ 第 2步合并错误，期望 ('e', 'st')，实际 {merges[1] if len(merges)>1 else 'None'}")
