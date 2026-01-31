import regex as re
from typing import List, Optional, Iterable, Iterator, Dict, Tuple

# GPT-2 的预分词正则表达式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        初始化 BPE tokenizer。

        Args:
            vocab: 词表，映射 token ID → bytes
            merges: BPE 合并规则列表，按创建顺序排列
            special_tokens: 特殊 token 列表，这些 tokens 不会被拆分
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 构建 bytes → token ID 的反向映射
        self.decoder = {v: k for k, v in vocab.items()}

        # 构建用于快速查找的 merge 字典
        # merge_map[(token_a, token_b)] = merged_token_id
        self.merge_map = {}
        for token_a, token_b in merges:
            merged = token_a + token_b
            self.merge_map[(token_a, token_b)] = self.decoder[merged]

        # 处理特殊 tokens
        self.special_token_ids = {}
        for token_str in self.special_tokens:
            token_bytes = token_str.encode("utf-8")
            if token_bytes in self.decoder:
                self.special_token_ids[token_str] = self.decoder[token_bytes]

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token ID 列表。

        Args:
            text: 要编码的文本

        Returns:
            token ID 列表
        """
        tokens = []

        # 先按特殊 tokens 切分
        if self.special_tokens:
            # 按最长的特殊 token 优先匹配（贪婪匹配）
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            pattern = '|'.join(re.escape(token) for token in sorted_special)
            segments = re.split(f'({pattern})', text)
        else:
            segments = [text]

        for segment in segments:
            # 如果是特殊 token，直接添加其 ID
            if segment in self.special_token_ids:
                tokens.append(self.special_token_ids[segment])
                continue

            # 如果是空字符串，跳过
            if not segment:
                continue

            # 使用 GPT-2 正则进行预分词
            words = re.findall(GPT2_SPLIT_PATTERN, segment)

            for word in words:
                if not word:
                    continue
                # 将单词转为 bytes，然后进行 BPE 编码
                word_bytes = word.encode("utf-8")
                tokens.extend(self._encode_bytes(word_bytes))

        return tokens

    def _encode_bytes(self, word_bytes: bytes) -> List[int]:
        """
        对单个 word_bytes 进行 BPE 编码。

        使用贪婪合并：优先合并 merges 列表中先出现的 pair。
        """
        if len(word_bytes) == 0:
            return []

        # 初始时，每个字节是一个 token
        parts = [bytes([b]) for b in word_bytes]

        # 贪婪合并：按照 merges 列表中的顺序
        # merges 列表是按优先级排序的（先添加的优先级高）
        while len(parts) > 1:
            # 在所有可能的合并中，选择优先级最高的（merges 列表中先出现的）
            best_pair = None
            best_pair_index = -1
            best_priority = len(self.merges)  # 默认优先级最低

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.merge_map:
                    # 找到这个 pair 在 merges 列表中的索引
                    # 索引越小，优先级越高
                    for priority, (p1, p2) in enumerate(self.merges):
                        if p1 == pair[0] and p2 == pair[1]:
                            if priority < best_priority:
                                best_priority = priority
                                best_pair = pair
                                best_pair_index = i
                            break

            if best_pair is None:
                break

            # 执行合并
            merged_bytes = best_pair[0] + best_pair[1]
            parts[best_pair_index:best_pair_index+2] = [merged_bytes]

        # 将 parts 转为 token IDs
        return [self.decoder[p] for p in parts]

    def encode_iterable(self, iterable: Iterable) -> Iterator[int]:
        """
        流式编码，适用于处理大文件。

        逐块读取输入并产生 token IDs，避免将整个文件加载到内存。

        Args:
            iterable: 可迭代对象，可以是文件对象或字符串列表

        Yields:
            token IDs
        """
        buffer = ""

        for chunk in iterable:
            # 如果是文件对象，chunk 是一行
            # 如果是字符串迭代器，chunk 是一个字符串
            buffer += chunk

            # 处理 buffer 中的完整 tokens
            # 我们需要确保不会在特殊 token 中间切断
            # 简单策略：按特殊 token 切分

            if self.special_tokens:
                sorted_special = sorted(self.special_tokens, key=len, reverse=True)
                pattern = '|'.join(re.escape(token) for token in sorted_special)

                # 查找所有特殊 token 的位置
                for match in re.finditer(pattern, buffer):
                    # 处理特殊 token 之前的部分
                    before_special = buffer[:match.start()]
                    if before_special:
                        words = re.findall(GPT2_SPLIT_PATTERN, before_special)
                        for word in words:
                            if word:
                                word_bytes = word.encode("utf-8")
                                for token_id in self._encode_bytes(word_bytes):
                                    yield token_id

                    # 产出特殊 token
                    special_token = match.group()
                    if special_token in self.special_token_ids:
                        yield self.special_token_ids[special_token]

                    # 更新 buffer
                    buffer = buffer[match.end():]

                # 处理剩余部分（不包含特殊 token）
                if buffer:
                    words = re.findall(GPT2_SPLIT_PATTERN, buffer)
                    for word in words[:-1]:  # 处理除最后一个外的所有单词
                        if word:
                            word_bytes = word.encode("utf-8")
                            for token_id in self._encode_bytes(word_bytes):
                                yield token_id
                    if words:
                        buffer = words[-1] if words else ""
                    else:
                        buffer = ""
            else:
                # 没有特殊 token，直接处理
                words = re.findall(GPT2_SPLIT_PATTERN, buffer)
                for word in words[:-1]:
                    if word:
                        word_bytes = word.encode("utf-8")
                        for token_id in self._encode_bytes(word_bytes):
                            yield token_id
                if words:
                    buffer = words[-1] if words else ""

        # 处理剩余的 buffer
        if buffer:
            words = re.findall(GPT2_SPLIT_PATTERN, buffer)
            for word in words:
                if word:
                    word_bytes = word.encode("utf-8")
                    for token_id in self._encode_bytes(word_bytes):
                        yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        将 token ID 列表解码为文本。

        Args:
            ids: token ID 列表

        Returns:
            解码后的文本
        """
        # 简单地将每个 ID 对应的 bytes 拼接
        result_bytes = b""
        for token_id in ids:
            if token_id in self.vocab:
                result_bytes += self.vocab[token_id]

        return result_bytes.decode("utf-8", errors="replace")
