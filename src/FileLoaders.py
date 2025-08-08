import copy
from pathlib import Path
import uuid
from loguru import logger
import os
import json
from .ConfigManager import config
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json


# 环境变量设置
os.environ['MINERU_MODEL_SOURCE'] = "local"

style_id_mapping = {
    "Normal": 0,
    "Heading1": 1,
    "Heading2": 2,
    "Heading3": 3,
    "Heading4": 4,
    "Heading5": 5,
}




# 文件处理函数
def file_type_identifier(file_path) -> str:
    """
    对输入的文件进行类型区分的函数
    """
    # 根据传入的path类型决定处理方式
    if isinstance(file_path, dict):
        return "quesion-answer"  # 如果传入dict，视为问答对
    else:
        # 获取文件后缀
        file_extension = os.path.splitext(file_path)[1].lower()

        # 匹配预设的文件类型，并返回
        if file_extension in config.get_setting("files")["types"]["text_file_extension"]:
            return "text"
        elif file_extension in config.get_setting("files")["types"]["document_file_extension"]:
            return "document"
        elif file_extension in config.get_setting("files")["types"]["audio_file_extension"]:
            return "audio"
        elif file_extension in config.get_setting("files")["types"]["table_file_extension"]:
            return "table"
        elif file_extension in config.get_setting("files")["types"]["image_file_extension"]:
            return "image"
        else:
            return "other"



### pdf 解析函数
def pdf_parse(
    file_type,  # 文件类型
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    p_formula_enable=True,  # Enable formula parsing
    p_table_enable=True,  # Enable table parsing
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,
    f_dump_content_list=True,
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    # 生成uuid
    time_based_id = str(uuid.uuid1())

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        pdf_bytes_list[idx] = new_pdf_bytes

    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=p_formula_enable,table_enable=p_table_enable)

    for idx, model_list in enumerate(infer_results):
        # model_json = copy.deepcopy(model_list)
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

        if file_type == 'document':
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
        
        if file_type == "image":
            image_dir = str(os.path.basename(local_image_dir))
            content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
            return content_list

        # if f_dump_middle_json:
        #     md_writer.write_string(
        #         f"{pdf_file_name}_middle.json",
        #         json.dumps(middle_json, ensure_ascii=False, indent=4),
        #     )

        logger.info(f"local output dir is {local_md_dir}")
        return {'text_path': os.path.join(local_md_dir, f"{pdf_file_name}.md"),
                "content_json_path": os.path.join(local_md_dir, f"{pdf_file_name}_content_list.json"),
                "image_path": local_image_dir,
                "temp_file_dir": os.path.join(output_dir, time_based_id),
                "temp_dir_path":local_md_dir}


def pdf_loader(
        file_path: Path,
        output_dir,
        lang="ch",
        method="auto",
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None  # End page ID for parsing, default is None (parse all pages until the end of the document)
):
    """
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        file_type = file_type_identifier(file_path)
        file_name = str(Path(file_path).stem)
        pdf_bytes = read_fn(file_path)
        file_name_list.append(file_name)
        pdf_bytes_list.append(pdf_bytes)
        lang_list.append(lang)
        extraction_info = pdf_parse(
                file_type=file_type,
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                parse_method=method,
                start_page_id=start_page_id,
                end_page_id=end_page_id
            )
        return extraction_info
    except Exception as e:
        logger.exception(e)


### docx 解析函数
import os
import json
import docx
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.enum.shape import WD_INLINE_SHAPE_TYPE
from docx.oxml.ns import qn 
import hashlib
import html


def get_para_format_profile(para: Paragraph) -> tuple:
    graph_info = {}
    p_fmt = para.paragraph_format
    for run in para.runs:
        font_name, font_size, font_bold, font_italic = None, None, None, None
        if run.text.strip():
            font = run.font
            font_name = font.name
            font_size = font.size.pt if font.size else None
            font_bold = font.bold
            font_italic = font.italic

            format_info = str(font_name) + '-' + str(font_size) + "-" + str() + "-" + str(font_bold) + str(font_italic)
            format_hash = hashlib.sha256(format_info.encode('utf-8')).hexdigest()
            run_count = len(run.text)
            if format_hash not in graph_info:
                graph_info[str(format_hash)] = run_count
            else:
                graph_info[str(format_hash)] += run_count
    return graph_info

def analyze_paragraphs(doc: Document, paragraph: Paragraph, image_save_path: str) -> dict:
    """
    对段落对象进行处理：
    1. 检查和保存图片
    2. 如果是'Normal'样式，统计各格式的字数
    3. 获取段落全文
    """
    # 初始化结果字典
    if paragraph is None:
        print(paragraph)

    result = {
        "full_text": "",
        "element_type": "garagraph",
        "image_paths": [],
        "paragraph_style_id": paragraph.style.style_id,
        "format_info": None
    }

    # 检查和保存图片
    for run in paragraph.runs:
        if run.element.find(qn('w:drawing')) is not None:
            r_embed_xpath = './/a:blip/@r:embed'
            r_id = run.element.xpath(r_embed_xpath)
            if r_id:
                image_r_id = r_id[0]
                # 通过 rId 从文档的关系部分获取图片“部件”
                image_part = doc.part.rels[image_r_id].target_part
                
                # 获取图片的二进制数据
                image_blob = image_part.blob
                
                # 生成一个唯一的文件名并保存
                unique_filename = f"{uuid.uuid4()}.jpg"
                save_path = os.path.join(image_save_path, unique_filename)
                
                with open(save_path, "wb") as f:
                    f.write(image_blob)
                
                result["image_paths"].append(save_path)

    # 样式检查和处理
    if paragraph.style.name in ['Normal', '正文']:
        graph_format_info = get_para_format_profile(paragraph)
        result["format_info"].append(graph_format_info)

    # 获取段落全文
    result["full_text"] = paragraph.text
    return result

def table_to_simple_html(table: Table):
    """
    将表格对象内容转为html格式的函数
    """
    html_lines = ["<table>"]

    # 检查表格是否为空
    if not table.rows:
        html_lines.append("</table>")
        return "\n".join(html_lines)

    # 将第一行作为表头 (<th>)，这对于很多解析器很重要
    header_row = table.rows[0]
    html_lines.append("  <thead>")
    html_lines.append("    <tr>")
    for cell in header_row.cells:
        # 清理并转义文本
        cell_text = html.escape(cell.text.strip())
        html_lines.append(f"      <th>{cell_text}</th>")
    html_lines.append("    </tr>")
    html_lines.append("  </thead>")

    # 处理表格主体数据 (<td>)
    html_lines.append("  <tbody>")
    for row in table.rows[1:]:
        html_lines.append("    <tr>")
        for cell in row.cells:
            cell_text = html.escape(cell.text.strip())
            html_lines.append(f"      <td>{cell_text}</td>")
        html_lines.append("    </tr>")
    html_lines.append("  </tbody>")
    
    html_lines.append("</table>")

    table_text = "\n".join(html_lines)
    return {"full_text": table_text, "element_type": "table", "paragraph_style_id": None}

def docx_to_content_list(docx_path: str, output_dir: str = 'output_images') -> dict:
    """
    解析docx文件结构的函数
    """
    # 读取文件
    document = Document(docx_path)
    doc_body_elements = document.element.body

    # 提取文件名
    file_name_with_ext = os.path.basename(docx_path)
    file_name_without_ext = os.path.splitext(file_name_with_ext)[0]

    # 参数准备
    image_output_dir = os.path.join(output_dir, 'images')
    final_content_list = []  # 最终的content_list
    first_round_content_list = []  # 第一轮循环的content_list
    normal_hash_count = {}  # 统计各种格式hash的影响
    p_tag = qn('w:p')       # 段落
    tbl_tag = qn('w:tbl')     # 表格

    # 生成保存目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    # 第一次结构循环，记录段落信息和格式信息
    for element in doc_body_elements:
        if element.tag == p_tag:
            print('element_type:', type(element))
            object_info = analyze_paragraphs(document, element, image_output_dir)

            if object_info["format_info"]:
                for info in object_info["format_info"]:
                    for hash, count in info.items():
                        if hash not in normal_hash_count:
                            normal_hash_count[hash] = count
                        else:
                            normal_hash_count[hash] += count 
        elif element.tag == tbl_tag:
            object_info == table_to_simple_html(element)
        else:
            print(f'忽视除表格和段落以外的元素类型')
            continue
        first_round_content_list.append(object_info)

    # 获取正文样式中，最常见的格式hash
    normal_format_hash = max(normal_hash_count, key=normal_hash_count.get)

    # 第二次循环，拼接和抽取信息，生成最终返回的content_lsit
    for idx, block in enumerate(first_round_content_list):
        if block['element_type'] == 'garagraph':
            block_image_path = block["image_paths"]
            block_style_id = block['paragraph_style_id']
            if len(block_image_path):
                # 提取图片信息
                start_idx = max(0, idx-2)
                image_caption = []
                image_footnote = []
                caption_flag = True
                footnote_flag = False

                for temp_block in first_round_content_list[start_idx:]:
                    # 获取image caption
                    if temp_block['paragraph_style_id'] == 'Caption' and caption_flag:
                        image_caption.append(temp_block['full_text'])
                    else:
                        caption_flag = False

                    # 获取image footnote
                    if temp_block['paragraph_style_id'] == 'FootnoteText':
                        image_footnote.append(temp_block['full_text'])
                        footnote_flag = True
                    
                    # 提前结束循环
                    if temp_block['paragraph_style_id'] != 'FootnoteText' and footnote_flag:
                        break
                    if len(temp_block["image_paths"]) == 0 and caption_flag is False:
                        footnote_flag = True

                if block["full_text"] and block_style_id not in ['Caption', 'FootnoteText']:
                    image_footnote.append(block["full_text"])


                # 保存图片信息
                for image_path in block["image_paths"]:
                    final_content_list.append({
                        "type": 'image',
                        "img_path": image_path,
                        "img_caption": image_caption,
                        "img_footnote": image_footnote
                    })
            else:
                # 提取文本信息
                full_text = block["full_text"]
                if block_style_id in style_id_mapping:
                    text_level = style_id_mapping[block_style_id]
                else:
                    text_level = 0

                if text_level:  # 非正文样式的文本，直接保存信息
                    final_content_list.append({
                        "type": "text",
                        "text": full_text,
                        "text_level": text_level
                    })
                else:  # 对于正文文本，检查具体格式信息: 当正文格式文字量大于50%， 视为正文文本
                    normal_count = 0
                    other_count = 0
                    for hash, count in block["format_info"].items():
                        if hash == normal_format_hash:
                            normal_count += count
                        else:
                            other_count += count
                    if normal_count / (normal_count + other_count) > 0.5:
                        final_content_list.append({
                            "type": "text",
                            "text": full_text,
                            "text_level": -1
                        })
                    else:
                        final_content_list.append({
                            "type": "text",
                            "text": full_text,
                            "text_level": 0
                        })
        elif block['element_type'] == 'table':
            # 类似图片类型处理
            start_idx = max(0, idx-2)
            table_caption = []
            table_footnote = []
            caption_flag = True
            footnote_flag = False
            block_style_id = block['paragraph_style_id']

            for temp_block in first_round_content_list[start_idx:]:
                # 获取image caption
                if temp_block['paragraph_style_id'] == 'Caption' and caption_flag:
                    table_caption.append(temp_block['full_text'])
                else:
                    caption_flag = False

                # 获取image footnote
                if temp_block['paragraph_style_id'] == 'FootnoteText':
                    table_footnote.append(temp_block['full_text'])
                    footnote_flag = True
                
                # 提前结束循环
                if temp_block['paragraph_style_id'] != 'FootnoteText' and footnote_flag:
                    break
                if len(temp_block["image_paths"]) == 0 and caption_flag is False:
                    footnote_flag = True

            # 保存表格信息
            for image_path in block["image_paths"]:
                final_content_list.append({
                    "type": 'image',
                    "table_body": block["full_text"],
                    "table_caption": table_caption,
                    "table_footnote": table_footnote
                })

        else:
            print(f'忽视除表格和段落以外的元素类型')
            continue
    
    # 保存最终的content_list
    json_name = f'{file_name_without_ext}_content_list.json'
    json_path = os.path.join(output_dir, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_content_list, f, ensure_ascii=False, indent=4)




### pptx 解析函数

if __name__ == '__main__':

    file_path = '/home/carlos/Projects/SmartAgent/data/Documents/Document.docx'


    # result = docx_to_content_list(file_path, output_dir='./upload')
    # print(result)

    file_path = '/home/carlos/Projects/SmartAgent/data/images/test_01.jpg'

    pdf_loader(file_path=file_path,
               output_dir='./output')