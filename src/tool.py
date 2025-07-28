# 多个文件会用到的工具函数
from typing import List, Optional, Union
import time
import functools


def timing_decorator(func):
    """
    一个计算并打印函数执行时间的装饰器。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        内部包装函数，用于实际执行并计时。
        """
        # 记录开始时间
        # time.perf_counter() 提供了高精度的性能计数器，适合测量短时间间隔
        start_time = time.perf_counter()
        
        # 执行被装饰的函数，并获取其返回值
        result = func(*args, **kwargs)
        
        # 记录结束时间
        end_time = time.perf_counter()
        
        # 计算耗时
        elapsed_time = end_time - start_time
        
        # 打印耗时信息
        # func.__name__ 会获取函数的实际名称
        print(f"函数 '{func.__name__}' 执行耗时: {elapsed_time:.4f} 秒")
        
        # 返回原函数的执行结果
        return result
        
    return wrapper


def standard_competition_ranking(values: list, reverse:bool=False):
    """
    标准竞争排名函数
    """
    sorted_scores = sorted(list(set(values)), reverse=reverse)  # 唯一值排序
    rank_dict = {}
    current_rank = 1

    for score in sorted_scores:
        rank_dict[score] = current_rank
        same_count = values.count(score)
        current_rank += same_count  # 跳过重复数量名次

    return [rank_dict[score] for score in values]







def fusion_list_str(content: Union[str, list]) -> str:
    """
    从list中拼接出字符串的函数
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        final_string = ""
        for text in content:
            if isinstance(text, str):
                final_string = final_string + text + '\n'
            else:
                final_string = final_string + str(text) + '\n'
    return final_string


def get_chunk_text(chunk, chunk_type):
    """
    提取图像和表格chunk中的信息
    """
    # 处理文本内容
    if chunk_type == "text":
        chunk_text = fusion_list_str(chunk["text"])  # 文本块的文本内容

    # 处理表格内容块
    if chunk_type == 'table':
        # 表格信息提取
        table_caption = fusion_list_str(chunk.get("table_caption", ""))
        table_footnote = fusion_list_str(chunk.get("table_footnote", ""))
        table_context = f"表格名称：: {table_caption}\n 表格尾注: {table_footnote}\n"
        table_content = fusion_list_str(chunk.get("table_body", ""))
        chunk_text = "表格名称：" + "\n" + table_caption + '\n' + "表格正文:" + "\n" + table_content + '\n' + "表格尾注信息：" + "\n" + table_footnote + '\n'

    # 处理图片内容块
    if chunk_type == 'image':
        # 图片信息提取
        img_caption = fusion_list_str(chunk.get("img_caption", ""))
        img_footnote = fusion_list_str(chunk.get("img_footnote", ""))
        chunk_text = "图片的名称信息:" + "\n" + img_caption + '\n' + "图片的尾注信息:" + img_footnote
    return chunk_text