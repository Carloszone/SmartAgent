import os
from jinja2 import Environment, FileSystemLoader
import json
import re
import textwrap
from typing import List
import numpy as np
import copy
from pathlib import Path
import uuid
from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

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
    elif mode == "text_correction":
        template = jinja2_env.get_template("AudioTextCorrection.jinja2")
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


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    p_formula_enable=True,  # Enable formula parsing
    p_table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-sglang-client backend
    f_draw_layout_bbox=False,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=False,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=False,  # Whether to dump middle JSON files
    f_dump_model_output=False,  # Whether to dump model output files
    f_dump_orig_pdf=False,  # Whether to dump original PDF files
    f_dump_content_list=False,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    # 生成uuid
    time_based_id = str(uuid.uuid1())

    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=p_formula_enable,table_enable=p_table_enable)

        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, time_based_id, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, p_formula_enable)

            pdf_info = middle_json["pdf_info"]

            pdf_bytes = pdf_bytes_list[idx]
            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4),
                )

            logger.info(f"local output dir is {local_md_dir}")
            return {'text_path': os.path.join(local_md_dir, f"{pdf_file_name}.md"),
                    "image_path": local_image_dir}
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        f_draw_span_bbox = False
        parse_method = "vlm"
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = vlm_doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url)

            pdf_info = middle_json["pdf_info"]

            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                md_writer.write_string(
                    f"{pdf_file_name}_model_output.txt",
                    model_output,
                )

            logger.info(f"local output dir is {local_md_dir}")
            return {'text_path': os.path.join(local_md_dir, f"{pdf_file_name}.md"),
                    "image_path": local_image_dir}


def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="ch",
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-sglang-engine: Faster(engine).
            vlm-sglang-client: Faster(client).
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        extraction_info = do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=backend,
                parse_method=method,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id
            )
        return extraction_info
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(__dir__)
    pdf_files_dir = os.path.join(__dir__, "pdfs")
    output_dir = os.path.join(project_root, "output")
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]

    doc_path_list = ['data/Documents/室内空气质量检测报告.pdf']
    for doc_path in Path(pdf_files_dir).glob('*'):
        if doc_path.suffix in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)

    """如果您由于网络问题无法下载模型，可以设置环境变量MINERU_MODEL_SOURCE为modelscope使用免代理仓库下载模型"""
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

    """Use pipeline mode if your environment does not support VLM"""
    result = parse_doc(doc_path_list, output_dir, backend="pipeline")
    print(result)