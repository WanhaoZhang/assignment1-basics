import os
import regex as re
from typing import Dict, List, Tuple
from collections import Counter

# GPT-2 的预分词正则表达式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    """
    
    # ==========================================
    # 1. 数据加载
    # ==========================================
    # TODO: 读取 input_path 对应的文件内容
    # 提示: 使用 open(input_path, "r", encoding="utf-8")
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()
    # print("ZHANG --------- "+text[:10])
    
    # ==========================================
    # 2. 预处理 (Pre-tokenization)
    # ==========================================
    # TODO: 实现 _pretokenize_and_count 函数
    # 提示：务必先处理 special_tokens，再进行正则切分
    word_counts = _pretokenize_and_count(text, special_tokens)
    # print("first time")
    # for word, freq in word_counts.items():
    #     print(f"{word}, {freq}\n")
            
    # ==========================================
    # 3. 初始化词表
    # ==========================================
    # 初始词表包含 256 个字节 (0-255)
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    
    # 计算需要执行多少次合并操作
    # 公式：目标大小 - 基础字符(256) - 特殊Token数量
    num_merges = vocab_size - 256 - len(special_tokens)
    merges: List[Tuple[bytes, bytes]] = []

    # ==========================================
    # 4. BPE 训练循环
    # ==========================================
    next_token_id = 256
    
    for i in range(num_merges):
        # 4.1 统计当前所有相邻字节对的频率
        # TODO: 实现 _get_pair_stats 函数
        pair_counts = _get_pair_stats(word_counts)
        
        if not pair_counts:
            break

        # 4.2 找到频率最高的 Pair
        # TODO: 实现 Tie-breaking 规则
        # 规则：频率最高优先；如果频率相同，选字典序更大的 (lexicographically greater)
        # 提示: max(pair_counts, key=lambda p: (...))
    
        best_pair = max(pair_counts, key=lambda p:(pair_counts[p], vocab[p[0]], vocab[p[1]]))
        
        # 4.3 记录合并规则
        # 提示：你需要从 vocab 中取出 best_pair 对应的 bytes，存入 merges 列表
        # token_bytes_a = ...
        # token_bytes_b = ...
        # merges.append(...)

        token_byte_a = vocab[best_pair[0]]
        token_byte_b = vocab[best_pair[1]]
        merges.append((token_byte_a, token_byte_b))
        
        # 4.4 更新词表
        # 将新生成的 token ID (next_token_id) 加入 vocab
        # new_token_bytes = ...
        # vocab[next_token_id] = new_token_bytes
        new_token_bytes = token_byte_a + token_byte_b
        vocab[next_token_id] = new_token_bytes
        
        # 4.5 更新统计数据
        # TODO: 实现 _merge_vocab 函数
        # 将所有出现 best_pair 的地方替换为 next_token_id
        word_counts = _merge_vocab(word_counts, best_pair, next_token_id)
        # print('for')
        # for word, freq in word_counts.items():
        #     print(f"{word}, {freq}\n")
        
        next_token_id += 1

    # ==========================================
    # 5. 添加 Special Tokens
    # ==========================================
    # TODO: 将 special_tokens 添加到词表末尾
    # 遍历 special_tokens，给它们分配 ID，并加入 vocab
    for token in special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1

    return vocab, merges


# ==========================================
# 辅助函数定义 (需要你来实现)
# ==========================================

def _pretokenize_and_count(text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    """
    先按 special_tokens 切分，再应用 GPT-2 正则，最后转为 bytes tuple 并统计频率。
    """
    counts = Counter()
    # TODO: 你的实现代码
    # 1. 使用 re.split 处理 special_tokens (建议用 re.escape)
    # 2. 对分出来的片段使用 re.findall(GPT2_SPLIT_PATTERN, ...)
    # 3. 将单词转为 UTF-8 bytes tuple 并存入 counts
    if special_tokens:
        pattern = '|'.join(re.escape(token) for token in special_tokens)
        segs = re.split(f"({pattern})", text)
    else:
        segs = [text]
    # print(segs)
    for seg in segs:
        if not seg or seg in special_tokens:
            continue
        
        words = re.findall(GPT2_SPLIT_PATTERN, seg)
        # print(words)
        for word in words:
            if not word: continue
            word_as_bytes = word.encode("utf-8")
            counts[tuple(word_as_bytes)] += 1
    # for i, (word, freq) in enumerate(counts.items()):
    #     print(f"{bytes(word).decode('utf-8')},{freq}")
    
    return counts

def _get_pair_stats(word_counts: Counter[tuple[int, ...]]) -> Dict[tuple[int, int], int]:
    """
    统计当前所有单词中相邻 token 对的出现频率。
    """
    pair_counts = Counter()
    # TODO: 你的实现代码
    # 遍历 word_counts 中的每个单词 (tuple of ints)
    # 统计相邻的 (item, next_item) 对
    for word, freq in word_counts.items():
        if len(word) < 2:
            continue
        
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq

    # for word_pair, freq in pair_counts.items():
    #     print(f"{word_pair}, {freq}")

    return pair_counts

def _merge_vocab(
    word_counts: Counter[tuple[int, ...]], 
    pair_to_merge: tuple[int, int], 
    new_token_id: int
) -> Counter[tuple[int, ...]]:
    """
    将 word_counts 中所有的 pair_to_merge 替换为 new_token_id。
    """
    new_counts = Counter()
    # TODO: 你的实现代码
    # 遍历 word_counts
    # 在单词中找到 pair_to_merge 并替换为 new_token_id
    # 提示: 如果单词太短或者不包含 pair 的第一个元素，可以直接跳过以优化速度
    for word, freq in word_counts.items():
        if len(word) < 2:
            new_counts[word] += freq
            continue
        if pair_to_merge[0] not in word:
            new_counts[word] += freq
            continue
        
        new_word = list(word)
        i = 0
        while i < len(new_word) - 1:
            cur_pair = (new_word[i], new_word[i+1])
            if cur_pair == pair_to_merge:
                new_word[i:i+2] = [new_token_id]
            else:
                i += 1
                
        new_counts[tuple(new_word)] += freq
    
    return new_counts


# ==========================================
# 测试代码 (根据 PDF 第 7 页的例子)
# ==========================================
if __name__ == "__main__":
    import tempfile
    import os
    
    print("="*40)
    print("  CS336 Assignment 1: BPE 全功能调试")
    print("="*40)

    # 1. 构造测试数据 
    # 基础数据：low(5), lower(2), newest(6), widest(3)
    # 特殊设计：
    #   我们在中间插入 "<|endoftext|>"。
    #   如果你的 regex/special token 处理正确，它应该被视为一个整体，且不影响前后单词的统计。
    special_token_str = "<|endoftext|>"
    
    # 构造文本
    # 注意：单词间加空格是为了让 GPT-2 regex 正确识别为单词
    base_text = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3
    input_text = base_text + special_token_str + " " + base_text # 翻倍数据量，夹杂特殊token
    
    print(f"[输入预览]: {input_text[:60]}...")
    print(f"[特殊Token]: {special_token_str}")

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(input_text)
        tmp_path = tmp_file.name

    try:
        # 2. 设置参数
        # 初始字节: 256 个
        # 计划 Merges: 3 次 (为了验证 st -> est -> ow/lo)
        # 特殊 Token: 1 个
        # 总 Vocab Size = 256 + 3 + 1 = 260
        target_vocab_size = 260
        
        print(f"\n[开始训练] 目标词表大小: {target_vocab_size}")
        
        vocab, merges = train_bpe(
            input_path=tmp_path,
            vocab_size=target_vocab_size,
            special_tokens=[special_token_str]
        )
        
        print("\n" + "-"*20 + " 训练结果 " + "-"*20)

        # 3. 验证 Merges (核心算法逻辑)
        print(f"生成的 Merges ({len(merges)}个):")
        for i, (b1, b2) in enumerate(merges):
            print(f"  {i+1}. {b1} + {b2}")

        # [cite: 196] 检查点 1: 频率最高且字典序最大
        # 's' + 't' (9+9=18次) vs 'e' + 's' (9+9=18次)
        # 字典序 't' > 's' (比较第二个元素)，或者整体比较 ('s', 't') > ('e', 's')
        # 预期冠军: ('s', 't') -> b'st'
        if len(merges) > 0:
            assert merges[0] == (b's', b't'), \
                f"❌ Merge 1 错误! 期望 (b's', b't'), 实际 {merges[0]} (检查字典序 Tie-breaking)"
            print("✅ Merge 1 通过: ('s', 't')")

        # [cite: 198] 检查点 2: 级联合并
        # 上一步生成了 b'st'。这一步应该是 'e' + b'st'
        if len(merges) > 1:
            assert merges[1] == (b'e', b'st'), \
                f"❌ Merge 2 错误! 期望 (b'e', b'st'), 实际 {merges[1]}"
            print("✅ Merge 2 通过: ('e', 'st')")

        # 检查点 3: 检查 'o' 和 'w'
        # 在 'low' 和 'lower' 中，'l'+'o' 和 'o'+'w' 频率相同。
        # 比较 tuple(b'l', b'o') vs tuple(b'o', b'w')
        # b'l'(108) < b'o'(111)。所以 ('o', 'w') 字典序更大，应该先合并。
        if len(merges) > 2:
            if merges[2] == (b'o', b'w'):
                 print("✅ Merge 3 通过: ('o', 'w') (字典序胜出)")
            elif merges[2] == (b'l', b'o'):
                 print("⚠️ Merge 3 是 ('l', 'o')。这在频率相同时是字典序较小的，请检查你的 max key 逻辑。")
            else:
                 print(f"❓ Merge 3 是 {merges[2]}")

        # 4. 验证 Special Token (ID 分配)
        # 逻辑：256 (base) + 3 (merges) = 259 个位置 (ID 0-258)
        # Special Token 应该是第 260 个位置 (ID 259)
        # [cite: 235] Special tokens do not affect BPE training (added at end)
        print("\n" + "-"*20 + " Special Tokens " + "-"*20)
        expected_st_id = 256 + 3  # = 259
        
        if expected_st_id in vocab:
            st_content = vocab[expected_st_id]
            print(f"ID {expected_st_id}: {st_content}")
            
            assert st_content == special_token_str.encode("utf-8"), \
                f"❌ Special Token 内容错误! 期望 {special_token_str.encode('utf-8')}, 实际 {st_content}"
            print("✅ Special Token 内容与位置正确")
        else:
            print(f"❌ 词表中找不到 ID {expected_st_id}，请检查 vocab_size 计算逻辑")

        # 5. 验证是否被错误切分
        # 检查 vocab 里是否有 '<' 或 '|' 这种被切碎的痕迹？
        # 简单检查：看 special token 的 ID 是否是独立分配的
        print("\n🎉 全功能测试完成！如果没有红色的❌，说明你的 BPE 训练器逻辑完美！")

    except NotImplementedError:
        print("\n⚠️  代码未完成 (NotImplementedError)")
    except AssertionError as e:
        print(f"\n❌ 断言失败: {e}")
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        if os.path.exists(tmp_path):
            os.remove(tmp_path)