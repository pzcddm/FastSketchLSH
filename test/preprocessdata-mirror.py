import argparse
import os
import time
from collections import Counter
from datasets import load_dataset


def download_and_prepare_dataset(dataset_name="pt-sk/pretraining-dataset",
                                 output_filename="pretraining_dataset_train_subset.txt",
                                 num_examples_to_process=50000):
    """
    从 Hugging Face 国内镜像下载数据集，并只处理 'train' 分块的一个子集以避免内存问题。
    """
    # 设置 Hugging Face 端点为国内镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print(f"Downloading dataset '{dataset_name}' from Hugging Face mirror...")
    try:
        # 正确加载数据集，它包含 train/validation/test 分块
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        print("Dataset downloaded successfully. Available splits:", list(dataset.keys()))

        # 从 'train' 分块中选择前 N 条记录进行处理，避免内存溢出
        print(f"Selecting the first {num_examples_to_process} examples from the 'train' split...")
        if 'train' not in dataset:
            raise ValueError("Dataset does not contain a 'train' split.")

        subset = dataset['train'].select(range(num_examples_to_process))

        print(f"Extracting data to '{output_filename}'...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            for item in subset:
                text_line = item['text'].replace('\n', ' ').strip()
                f.write(text_line + '\n')

        print(f"Dataset extraction complete. Input file '{output_filename}' is ready.")
        return output_filename

    except Exception as e:
        print(f"\nError: Failed to download or process dataset: {e}")
        print("Please ensure you have an internet connection and the required libraries installed:")
        print("pip install datasets huggingface_hub")
        return None


def process_data(input_file, output_file, sort_elements_desc, sort_sets_desc, dedup_items, dedup_sets, write_id_prefix,
                 set_type):
    """
    使用 Python 复现 C++ 数据处理逻辑。
    """
    str_map_id = {}
    id_counter = Counter()
    sets = []

    if dedup_sets:
        pre_sets_unique = set()

    # --- 阶段 1: 读取文件，将字符串映射为初始 ID，并统计频率 ---
    print(f"Reading and processing '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            raw_elements = line.strip().split()
            if not raw_elements:
                continue
            current_set_ids = []
            if dedup_items:
                processed_elements = list(dict.fromkeys(raw_elements))
            else:
                line_element_counter = Counter()
                processed_elements = []
                for element in raw_elements:
                    unique_element_str = f"{element}{line_element_counter[element]}"
                    processed_elements.append(unique_element_str)
                    line_element_counter[element] += 1
            for element in processed_elements:
                if element not in str_map_id:
                    new_id = len(str_map_id)
                    str_map_id[element] = new_id
                element_id = str_map_id[element]
                current_set_ids.append(element_id)
            if dedup_sets:
                pre_sets_unique.add(frozenset(current_set_ids))
            else:
                sets.append(current_set_ids)
                for element_id in current_set_ids:
                    id_counter[element_id] += 1

    if dedup_sets:
        sets = [list(s) for s in pre_sets_unique]
        for s in sets:
            for element_id in s:
                id_counter[element_id] += 1

    # --- 阶段 2: 基于频率对 ID 进行重映射 ---
    sorted_ids = sorted(id_counter.keys(), key=lambda x: (id_counter[x], x))
    new_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}

    # --- 阶段 3: 应用新映射并排序 ---
    processed_sets = []
    for s in sets:
        new_set = [new_mapping[old_id] for old_id in s]
        new_set.sort(reverse=sort_elements_desc)
        processed_sets.append(new_set)
    processed_sets.sort(key=lambda s: (len(s), s), reverse=sort_sets_desc)

    # --- 阶段 4: 写入输出文件 ---
    print(f"Writing processed data to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, s in enumerate(processed_sets):
            line_parts = []
            if write_id_prefix:
                line_parts.append(f"{set_type}{i}")
            line_parts.extend(map(str, s))
            outfile.write(" ".join(line_parts) + "\n")


def main():
    # --- 步骤 1: 下载和准备数据 ---
    input_filename = "pretraining_dataset_train_subset.txt"
    if not os.path.exists(input_filename):
        # 您可以在这里调整要处理的数据量
        input_filename = download_and_prepare_dataset(
            output_filename=input_filename,
            num_examples_to_process=50000
        )
        if not input_filename:
            return
    else:
        print(f"Found existing dataset file: '{input_filename}'. Skipping download.")

    # --- 步骤 2: 解析命令行参数并执行数据处理 ---
    parser = argparse.ArgumentParser(
        description="使用 Python 实现的数据处理工具，功能与提供的 C++ 代码一致。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_file', nargs='?', default=input_filename,
                        help=f"输入文件的路径 (默认: {input_filename})")
    parser.add_argument('output_file',
                        help="输出文件的路径 (例如: processed_data.txt)")
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--inc1', dest='sort_elements_desc', action='store_false', help='按频率递增排序集合内元素')
    group1.add_argument('--dec1', dest='sort_elements_desc', action='store_true',
                        help='按频率递减排序集合内元素 (默认)')
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--inc2', dest='sort_sets_desc', action='store_false', help='按长度递增排序集合')
    group2.add_argument('--dec2', dest='sort_sets_desc', action='store_true', help='按长度递减排序集合 (默认)')
    parser.add_argument('--dedupitems', action='store_true', default=False, help='去除集合中重复的元素')
    parser.add_argument('--dedup', dest='dedup_sets', action='store_true', default=False, help='去除数据集中重复的集合')
    parser.add_argument('--wid', nargs=1, metavar='TYPE', help='在每行输出前添加 ID，并指定类型 (例如: r 或 s)')
    parser.set_defaults(sort_elements_desc=True, sort_sets_desc=True)
    args = parser.parse_args()

    set_type = args.wid[0] if args.wid else None

    print("\n--- Starting Data Processing ---")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Sort elements by frequency: {'decreasing' if args.sort_elements_desc else 'increasing'}")
    print(f"Sort sets by length: {'decreasing' if args.sort_sets_desc else 'increasing'}")
    if args.dedupitems: print("Deduplicate items within sets: ON")
    if args.dedup_sets: print("Deduplicate entire sets: ON")

    start_time = time.time()
    process_data(
        input_file=args.input_file,
        output_file=args.output_file,
        sort_elements_desc=args.sort_elements_desc,
        sort_sets_desc=args.sort_sets_desc,
        dedup_items=args.dedupitems,
        dedup_sets=args.dedup_sets,
        write_id_prefix=bool(args.wid),
        set_type=set_type
    )
    end_time = time.time()

    print("\nProcessing successful!")
    print(f"Total time taken: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()