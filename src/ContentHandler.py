import os
from jinja2 import Environment, FileSystemLoader
import json
import re
import textwrap
from typing import List
import numpy as np

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
    elif mode == 'content_pitch':  # 文本拼接模板
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
    else:
        raise ValueError(f'mode参数错误，当前参数为{mode}')
    message = textwrap.dedent(template.render())

    return message

def json_extractor(content: str):

    """
    用于从大模型返回的内容中提取json格式的函数
    :param content: 大模型返回的字符串
    :return:
    """
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


def query_result_handler(result):
    output_dcit = {}
    for i, object in enumerate(result.objects):
        uuid = object["uuid"]
        output_dcit[str(uuid)] = i


def apply_rrf(search_results: List, k:int = 60) -> dict:
    """
    对多路搜索结果进行RRF排序
    """
    # 提取记录的uuid信息
    all_uuids = []
    search_dict = {}
    for index, search_result in enumerate(search_results):
        search_dict[index] = {}
        for ind, search_object in enumerate(search_result.object):
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