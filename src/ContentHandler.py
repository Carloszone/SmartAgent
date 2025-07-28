import os
from jinja2 import Environment, FileSystemLoader
import json
import re
import textwrap
from typing import List, Optional, Union
import numpy as np
from langchain_community.document_loaders import TextLoader
from .FileLoaders import pdf_loader
from bs4 import BeautifulSoup
from .tool import timing_decorator

file_loader_mapping = {
            # 文档对象
            ".pdf": pdf_loader,
            # ".docx": docx_Loader,
            ".txt": TextLoader,
            ".md": TextLoader,
            ".text": TextLoader,
            ".ppt": "",

            # 表格对象
            ".csv": "",
            ".xls": "",

            # 图片对象
            ".jpg": "",
            ".png": ""
        }


def get_system_message(mode: str):
    """
    基于mode参数,生成message需要的system和user模板
    """
    # 定义环境
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    template_directory_path = os.path.join(project_root, 'data', 'PromptTemplate')
    jinja2_env = Environment(loader=FileSystemLoader(template_directory_path))

    if mode == 'markdown':  # markdown格式化模板
        template = jinja2_env.get_template('MarkdownTemplate.jinja2')
    elif mode == 'text_fusion':  # 文本拼接模板
        template = jinja2_env.get_template("WindowPitchTemplate.jinja2")
    elif mode == "chunker":  # 文本分块模板
        template = jinja2_env.get_template("ChunkSplitterTemplate.jinja2")
    elif mode == 'text_summary':  # 文本概括模板
        template = jinja2_env.get_template("TextSummaryTemplate.jinja2")
    elif mode == "keywords":  # 文本关键词提取模板
        template = jinja2_env.get_template("KeywordsTemplate.jinja2")
    elif mode == "questions":  # 文本提问模板
        template = jinja2_env.get_template("QuestionGenerationTemplate.jinja2")
    elif mode == "topic_extraction":  # 主题提取模板
        template = jinja2_env.get_template("TopicExtractionTemplate.jinja2")
    elif mode == "answer_summary":  # 回答摘要模板
        template = jinja2_env.get_template("AnswerSummaryTemplate.jinja2")
    elif mode == "text_correction":  # 文本纠偏模板
        template = jinja2_env.get_template("AudioTextCorrection.jinja2")
    elif mode == "table_fusion":  # 表格整合模板
        template = jinja2_env.get_template("TableFusionTemplate.jinja2")
    elif mode == "table_convert":  # 表格转化模板
        template = jinja2_env.get_template("Image2TableTemplate.jinja2")
    elif mode == "image_description":  # 图片描述融合模板
        template = jinja2_env.get_template("ImageDescriptionTemplate.jinja2")
    elif mode == "image_caption":  # 图片内容捕捉模板
        template = jinja2_env.get_template("ImageCaptionTemplate.jinja2")
    elif mode == 'hyde':  # HyDE回答生成
        template = jinja2_env.get_template('HyDETemplate.jinja2')
    else:
        raise ValueError(f'mode参数错误，当前参数为{mode}')
    message = textwrap.dedent(template.render())

    return message


def split_text_by_paragraphs(text, parag_per_chunk: int) -> List[str]:
    """
    根据段落将文本拆分的函数
    """
    # 检查text格式
    if not text:
        return []
    
    # 使用正则表达式分割成段落列表，并清除空段落
    paragraphs = re.split(r'(\n\s*){2,}', text)

    # 过滤掉分割后产生的空字符串或只包含空格的字符串
    non_empty_paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
    
    # 检查过滤后的文本list
    if not non_empty_paragraphs:
        return []
    
    # 遍历和提取list中的信息
    text_chunks = []
    for i in range(0, len(non_empty_paragraphs), parag_per_chunk):
        # 获取当前块的段落
        chunk_paragraphs = non_empty_paragraphs[i : i + parag_per_chunk]
        # 使用 `\n\n` 将段落重新连接成一个文本块，以保留段落结构
        text_chunk = "\n\n".join(chunk_paragraphs)
        text_chunks.append(text_chunk)
    return text_chunks
    

def json_extractor(content: str):

    """
    用于从大模型返回的内容中提取json格式的函数
    :param content: 大模型返回的字符串
    :return:
    """
    # 转义\检查
    pattern = re.compile(r'\\(?![\\"\/bfnrtu])')
    content = pattern.sub(r'\\\\', content)

    # 第一次尝试：大模型返回了纯净的json字符串
    try: 
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 第二次尝试，大模型返回的是markdown代码块
    try:
        match = re.search(r'```(json)?\s*(\{.*\}|\[.*\])\s*```', content, re.DOTALL)
        if match:
            json_str = match.group(2)
            return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        pass

    # 第三次尝试，寻找第一个{和最后一个}
    try:
        # 尝试寻找JSON对象
        start_brace = content.find('{')
        end_brace = content.rfind('}')
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            json_str = content[start_brace : end_brace + 1]
            # print('3th :',json_str)
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 第四次尝试，寻找第一个[和最后一个]
    try:
        start_bracket = content.find('[')
        end_bracket = content.rfind(']')
        if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
            json_str = content[start_bracket : end_bracket + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass 

    raise ValueError(f"无法从输入中提取JSON数据,当前的输入信息为：{content}")


def find_files_in_directory(directory_path: str, target_extension: Optional[str] = None) -> List:
    """
    检查指定的文件夹中是否有文件。

    参数:
    directory_path (str): 要检查的文件夹路径。

    返回:
    list: 如果文件夹中存在文件，则返回包含所有文件完整路径的列表。
    None: 如果文件夹不存在、不是一个有效的目录，或者文件夹为空（没有任何文件）。
    """
    # 检查路径是否存在以及是否是一个目录
    if not os.path.isdir(directory_path):
        print(f"错误：提供的路径 '{directory_path}' 不是一个有效的目录或不存在。")
        return None

    # 获取目录下的所有条目
    all_entries = os.listdir(directory_path)

    # 后缀预处理
    if target_extension:
    # 确保后缀以点开头
        if not target_extension.startswith('.'):
            target_extension = '.' + target_extension
        # 将后缀转为小写，以进行不区分大小写的比较
        target_extension = target_extension.lower()
    
    # 构建所有条目的完整路径，并筛选出文件
    found_files = []
    try:
        # 3. 遍历目录下的所有条目
        for entry_name in all_entries:
            full_path = os.path.join(directory_path, entry_name)
        
            # 检查当前条目是否为文件
            if os.path.isfile(full_path):
            # 4. 如果指定了后缀，则进行筛选
                if target_extension:
                    # 使用 .lower() 来进行不区分大小写的后缀名比较
                    if entry_name.lower().endswith(target_extension):
                        found_files.append(full_path)
            else:
                # 如果没有指定后缀，则添加所有文件
                found_files.append(full_path)

        return found_files
    except OSError as e:
        print(f"访问目录时发生错误: {e}")
    return []


def get_context_around_image(file_path: str, image_name: str, num_paragraphs: int = 2) -> str:
    """
    从 Markdown 文件中截取包含特定图片行的上下文段落。

    Args:
        file_path (str): .md 文件的路径。
        image_name (str): 要查找的图片文件名 (例如 '68a06be6...jpg')。
        num_paragraphs (int, optional): 上下文要包含的段落数 (N)。默认为 2。

    Returns:
        str: 包含上下文的文本字符串。如果找不到文件或图片，则返回错误信息。
    """
    # 1. 读取文件内容
    if not os.path.exists(file_path):
        return f"错误：文件不存在 -> {file_path}"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. 定义“段落”：使用正则表达式按一个或多个空行分割文本
    #    这种方法比简单的 split('\n\n') 更健壮
    paragraphs = re.split(r'\n\s*\n', content)
    
    # 清理可能因分割产生的空白段落
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 3. 查找包含目标图片名的段落索引
    target_index = -1
    for i, p in enumerate(paragraphs):
        if image_name in p:
            target_index = i
            break

    if target_index == -1:
        print(f'警告：在文档中未找到图片 -> {image_name}')
        return {"get_state": False, "get_content": ""}

    # 4. 计算上下文的起止索引
    #    使用 max(0, ...) 来处理前面段落不足 N 个的情况
    start_index = max(0, target_index - num_paragraphs)
    
    # 结束索引。Python 的切片是右开区间，所以 +1
    end_index = target_index + num_paragraphs + 1

    # 5. 提取上下文段落（Python 的列表切片会自动处理末尾越界的情况）
    context_paragraphs = paragraphs[start_index:end_index]

    # 6. 将提取的段落重新组合成一个字符串并返回
    # return '\n\n'.join(context_paragraphs)
    return {"get_state": True, "get_content": '\n\n'.join(context_paragraphs)}


def find_html_tables_in_markdown(file_path: str) -> List[str]:
    """
    在指定的Markdown文件中查找所有嵌入的HTML表格块。

    表格的格式必须是 "<html><body><table>...</table></body></html>"。

    Args:
        file_path (str): 要搜索的 .md 文件的路径。

    Returns:
        List[str]: 一个列表，其中每个元素都是一个找到的完整HTML表格字符串。
                   如果文件不存在或未找到表格，则返回空列表。
    """
    # 检查文件是否存在，避免程序因找不到文件而崩溃
    if not os.path.exists(file_path):
        print(f"警告：文件不存在 -> {file_path}")
        return []

    try:
        # 使用 'with' 语句安全地读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

    # 定义正则表达式模式来匹配整个HTML表格块
    # - r'...' 表示这是一个原始字符串，可以避免反斜杠问题
    # - <html><body><table> 和 </table></body></html> 是字面匹配的起始和结束标记
    # - .*? 是核心部分：
    #   - .  匹配除换行符以外的任何字符
    #   - * 匹配前面的字符零次或多次
    #   - ?  使匹配变为“非贪婪模式”，即匹配尽可能少的字符，确保在有多个表格时，它会在第一个 "</table>" 处停止，而不是匹配到最后一个
    # - re.DOTALL 标志是一个关键，它允许 '.' 匹配包括换行符在内的所有字符，因为表格HTML会跨越多行
    pattern = r'<html><body><table>.*?</table></body></html>'
    
    # 使用 re.findall 找到所有不重叠的匹配项，并以列表形式返回
    found_tables = re.findall(pattern, content, re.DOTALL)
    
    return found_tables


def apply_rrf(search_results: List, k:int = 60) -> dict:
    """
    对多路搜索结果进行RRF排序
    """
    # 提取记录的uuid信息
    all_uuids = []
    search_dict = {}
    for index, search_result in enumerate(search_results):
        search_dict[index] = {}
        for ind, search_object in enumerate(search_result.objects):
            uuid = search_object.uuid
            all_uuids.append(uuid)
            search_dict[index][uuid] = ind + 1

    # 对记录的uuid列表进行去重操作
    unique_uuids = list(set(all_uuids))

    # 计算每个记录的融合分数
    rrf_scores = {}
    for uuid in all_uuids:
        temp_scores = []
        for key, value in search_dict.items():
            if uuid in value.keys():
                rrf_score =  1 / (value[uuid] + k)
            else:
                rrf_score = 0
            temp_scores.append(rrf_score)
        rrf_scores[uuid] = np.mean(temp_scores)
            
    # 对所有记录进行降序排序
    sorted_uuids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return sorted_uuids


def html_to_json(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    is_table = soup.find('table')

    table_data = []
    if is_table:
        rows = is_table.find_all('tr')
        for row in rows:
            cols = row.find_all(['td', 'th'])
            cols_text = [ele.text.strip() for ele in cols]
            table_data.append(cols_text)

        original_json = json.dumps(table_data, indent=2, ensure_ascii=False)

        # 表格内容处理
        table_headers = table_data[0]
        table_rows = table_data[1:]
        table_list = []
        for row in table_rows:
            obj = {table_headers[i]: (row[i] if i < len(row) else None) for i in range(len(table_headers))}
            table_list.append(obj)

        table_json = json.dumps(table_list, indent=2, ensure_ascii=False)
        
        return table_json
    else:
        print('未能找到表格信息')
        return ''

@timing_decorator
def file_loader(file_path, output_dir: str=None) -> dict:
    """
    读取文件的读取器
    目前支持的文件类型有：.pdf, .txt, .md
    """

    # 获取文件后缀
    file_extension = os.path.splitext(file_path)[1].lower()

    # 进行后缀匹配
    if file_extension in file_loader_mapping:
        document_loader = file_loader_mapping.get(file_extension)

        # 基于后缀，创建loader实例
        if file_extension in file_loader_mapping.keys():
            if output_dir:
                file_info = document_loader(file_path, output_dir=output_dir)
            else:
                file_info = document_loader(file_path)
                # print(f"返回的变量类型：{type(file_info)}")

        # 加载并返回文档内容
        return file_info
    else:
        print(f"警告：不支持的文件类型 {file_extension}，跳过文件 {file_path}")
        return {}




if __name__ == '__main__':
    # parse_result = file_loader("/home/carlos/Projects/SmartAgent/data/Documents/2025数据分析Agent实践与案例研究报告.pdf", "./output")
    # print(parse_result)

    test_str = """
```json
{
  "report": "表1主要检测仪器一览表，包含3列：序号、检测仪器名称、型号和仪器编号，共2行数据。"
}
```
"""
    output_content = json_extractor(test_str)
    print(output_content)
