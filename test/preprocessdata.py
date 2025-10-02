import argparse
import time
from collections import defaultdict, Counter
from typing import List, Set, Dict, Tuple
from datasets import load_dataset
import os


class DatasetProcessor:
    def __init__(self):
        self.pre_sets = set()
        self.sets = []
        self.sets_ID_num = []
        self.str_map_id = {}
        self.new_mapping = {}
        self.elements_num = 0

        # 默认参数
        self.flag1 = True  # 集合元素排序：True为降序，False为升序
        self.flag2 = True  # 集合排序：True为降序，False为升序
        self.dedupitems = False  # 去除集合中重复元素
        self.dedup = False  # 去除重复集合
        self.wid = False  # 是否写入集合ID
        self.settype = 'r'  # 集合ID类型

    def cmp_key(self, item):
        """比较函数：按元素频率升序排列，如果频率相同，则按元素值升序排列"""
        return (item[1], item[0])

    def download_huggingface_dataset(self, dataset_name: str, split: str = 'train',
                                     text_column: str = 'text', output_file: str = None):
        """
        从Hugging Face下载数据集并保存为文本文件

        Args:
            dataset_name: 数据集名称，如 'wikitext', 'bookcorpus' 等
            split: 数据集分割，如 'train', 'test', 'validation'
            text_column: 包含文本的列名
            output_file: 输出文件路径
        """
        print(f"正在下载数据集: {dataset_name}")

        try:
            # 加载数据集
            dataset = load_dataset(dataset_name, split=split)
            print(f"数据集加载成功，共有 {len(dataset)} 条数据")

            # 保存为文本文件
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, example in enumerate(dataset):
                    text = example[text_column]
                    if isinstance(text, str):
                        f.write(text + '\n')
                    # 如果是列表，则用空格连接
                    elif isinstance(text, list):
                        f.write(' '.join(str(x) for x in text) + '\n')

                    if (i + 1) % 10000 == 0:
                        print(f"已处理 {i + 1} 条数据")

            print(f"数据集已保存到: {output_file}")
            return output_file

        except Exception as e:
            print(f"下载数据集时出错: {e}")
            return None

    def process_line_to_tokens(self, line: str, ngram: int = 1) -> List[str]:
        """
        将一行文本转换为tokens（可以扩展为n-gram）

        Args:
            line: 输入文本行
            ngram: n-gram大小，1表示单词，2表示2-gram等
        """
        # 基本分词（按空格）
        tokens = line.strip().split()

        # 如果需要n-gram，可以在这里实现
        if ngram > 1:
            ngram_tokens = []
            for i in range(len(tokens) - ngram + 1):
                ngram_tokens.append('_'.join(tokens[i:i + ngram]))
            return ngram_tokens

        return tokens

    def process_input(self, input_file: str, output_file: str, ngram: int = 1):
        """处理输入文件"""

        print("开始处理数据...")

        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # 第一遍：处理所有行，构建元素映射
        for line_num, line in enumerate(lines):
            tokens = self.process_line_to_tokens(line, ngram)
            set_element_num = defaultdict(int)
            st = set()
            vec = []

            for element in tokens:
                # 去重逻辑
                if self.dedupitems and set_element_num[element] > 0:
                    continue

                cnt = set_element_num[element]
                if cnt == 0:
                    set_element_num[element] = 1
                    unique_element = element + "0"
                else:
                    set_element_num[element] += 1
                    unique_element = element + str(cnt)

                # 将元素映射到全局ID
                if unique_element not in self.str_map_id:
                    self.str_map_id[unique_element] = self.elements_num
                    self.sets_ID_num.append([self.elements_num, 0])
                    self.elements_num += 1
                elif not self.dedup:
                    element_id = self.str_map_id[unique_element]
                    self.sets_ID_num[element_id][1] += 1

                element_id = self.str_map_id[unique_element]

                if self.dedup:
                    st.add(element_id)
                else:
                    vec.append(element_id)

            if self.dedup:
                # 使用frozenset作为可哈希的set
                self.pre_sets.add(frozenset(st))
            else:
                self.sets.append(vec)

            if (line_num + 1) % 10000 == 0:
                print(f"已处理 {line_num + 1} 行")

        # 如果需要去除重复集合，则统计各ID出现频率
        if self.dedup:
            for st in self.pre_sets:
                vec = list(st)
                for element_id in st:
                    self.sets_ID_num[element_id][1] += 1
                self.sets.append(vec)

        print(f"元素总数: {self.elements_num}")
        print(f"集合总数: {len(self.sets)}")

        # 按照出现频率排序
        self.sets_ID_num.sort(key=self.cmp_key)

        # 重新映射ID
        for new_id, (old_id, freq) in enumerate(self.sets_ID_num):
            self.new_mapping[old_id] = new_id

        # 重新映射集合中的元素
        for i in range(len(self.sets)):
            self.sets[i] = [self.new_mapping[old_id] for old_id in self.sets[i]]

        # 对集合中的元素排序
        for i in range(len(self.sets)):
            if self.flag1:  # 降序
                self.sets[i].sort(reverse=True)
            else:  # 升序
                self.sets[i].sort()

        # 对集合排序
        if self.flag2:  # 按长度降序
            self.sets.sort(key=lambda x: (-len(x), x))
        else:  # 按长度升序
            self.sets.sort(key=lambda x: (len(x), x))

        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i, set_items in enumerate(self.sets):
                if self.wid:
                    outfile.write(f"{self.settype}{i} ")
                outfile.write(' '.join(map(str, set_items)) + '\n')

        print("数据处理完成!")


def main():
    parser = argparse.ArgumentParser(description='处理Hugging Face数据集')
    parser.add_argument('dataset_name', help='Hugging Face数据集名称')
    parser.add_argument('output_file', help='输出文件路径')
    parser.add_argument('--split', default='train', help='数据集分割 (default: train)')
    parser.add_argument('--text_column', default='text', help='文本列名 (default: text)')
    parser.add_argument('--ngram', type=int, default=1, help='n-gram大小 (default: 1)')
    parser.add_argument('--cache_dir', help='缓存目录')

    # 处理选项
    parser.add_argument('--inc1', action='store_true', help='集合元素递增排序')
    parser.add_argument('--dec1', action='store_true', help='集合元素递减排序')
    parser.add_argument('--inc2', action='store_true', help='集合长度递增排序')
    parser.add_argument('--dec2', action='store_true', help='集合长度递减排序')
    parser.add_argument('--dedupitems', action='store_true', help='去除集合中重复元素')
    parser.add_argument('--dedup', action='store_true', help='去除重复集合')
    parser.add_argument('--wid', help='写入集合ID (r/s)')

    args = parser.parse_args()

    # 创建处理器实例
    processor = DatasetProcessor()

    # 设置处理选项
    if args.inc1:
        processor.flag1 = False
    if args.dec1:
        processor.flag1 = True
    if args.inc2:
        processor.flag2 = False
    if args.dec2:
        processor.flag2 = True
    if args.dedupitems:
        processor.dedupitems = True
    if args.dedup:
        processor.dedup = True
    if args.wid:
        processor.wid = True
        processor.settype = args.wid

    print(f"集合元素按频率{'递增' if not processor.flag1 else '递减'}排序")
    print(f"集合按长度{'递增' if not processor.flag2 else '递减'}排序")
    if processor.dedupitems:
        print("已开启去除集合中重复元素的功能")
    if processor.dedup:
        print("已开启去除重复集合的功能")

    start_time = time.time()

    # 临时文件路径
    temp_file = f"temp_{args.dataset_name.replace('/', '_')}_{args.split}.txt"

    # 下载并处理数据集
    try:
        # 下载数据集
        downloaded_file = processor.download_huggingface_dataset(
            args.dataset_name,
            args.split,
            args.text_column,
            temp_file
        )

        if downloaded_file:
            # 处理数据
            processor.process_input(downloaded_file, args.output_file, args.ngram)

            # 清理临时文件
            os.remove(temp_file)
            print(f"临时文件已删除: {temp_file}")

    except Exception as e:
        print(f"处理过程中出错: {e}")
    finally:
        # 确保临时文件被清理
        if os.path.exists(temp_file):
            os.remove(temp_file)

    end_time = time.time()
    print(f"预处理成功！总计耗时 {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    # 示例用法：
    # python script.py wikitext wikitext-103-v1 output.txt --split train --dedup --dec1
    main()