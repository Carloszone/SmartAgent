import copy
from pathlib import Path
import uuid
from loguru import logger
import os

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

from paddleocr import PaddleOCR

def pdf_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    p_formula_enable=True,  # Enable formula parsing
    p_table_enable=True,  # Enable table parsing
    f_dump_md=True,  # Whether to dump markdown files
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

        if f_dump_md:
            image_dir = str(os.path.basename(local_image_dir))
            md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
            md_writer.write_string(
                f"{pdf_file_name}.md",
                md_content_str,
            )

        logger.info(f"local output dir is {local_md_dir}")
        return {'text_path': os.path.join(local_md_dir, f"{pdf_file_name}.md"),
                "image_path": local_image_dir,
                "temp_dir_path":os.path.join(output_dir, time_based_id)}


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
        file_name = str(Path(file_path).stem)
        pdf_bytes = read_fn(file_path)
        file_name_list.append(file_name)
        pdf_bytes_list.append(pdf_bytes)
        lang_list.append(lang)
        extraction_info = pdf_parse(
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

def image_text_reader(image_path):
    ocr = PaddleOCR(use_textline_orientation=True, lang="ch")
    result = ocr.predict(image_path)
    for line in result:
        print(line["rec_texts"])


# if __name__ == '__main__':
#     __dir__ = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(__dir__)
#     pdf_files_dir = os.path.join(__dir__, "pdfs")
#     output_dir = os.path.join(project_root, "output")
#     pdf_suffixes = [".pdf", ".docx", ".pptx"]
#     image_suffixes = [".png", ".jpeg", ".jpg"]

#     doc_path_list = ['output/b99a556e-56e5-11f0-8351-74563c6e57c2/auto/images/f2b23d170b3bf85055e3e8664805df1c920cac7d885baabcfcc4e40c6f335cb1.jpg']
#     for doc_path in Path(pdf_files_dir).glob('*'):
#         if doc_path.suffix in pdf_suffixes + image_suffixes:
#             doc_path_list.append(doc_path)

#     """如果您由于网络问题无法下载模型，可以设置环境变量MINERU_MODEL_SOURCE为modelscope使用免代理仓库下载模型"""
#     # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

#     """Use pipeline mode if your environment does not support VLM"""
#     for path in doc_path_list:
#         result = pdf_loader(path, output_dir)
#         print(result)


import docx
import os
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

def docx_Loader(docx_path, output_folder):
    """
    将 DOCX 文件解析，处理并输出的函数。
    - 文本和标题将转换为 Markdown 语法。
    - 表格将转换为 HTML <table> 标签。
    - 图像将被提取到指定文件夹，并在 Markdown 中通过相对路径引用。
    输出格式：
    {'text_path': Markdown格式化的.md文件地址,
     "image_path": 保存图像的文件夹地址,
      "temp_dir_path": 保存元素内容的临时文件夹地址}
    """
    # 生成uuid
    time_based_id = str(uuid.uuid1())

    # 生成临时输出文件夹
    output_folder = os.path.join(output_folder, time_based_id)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 为图片创建一个子文件夹
    image_folder = os.path.join(output_folder, 'images')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    doc = docx.Document(docx_path)
    markdown_lines = []
    image_counter = 1

    # 创建一个 doc.tables 的迭代器，以便按顺序访问
    tables_iter = iter(doc.tables)

    # 遍历文档的顶级块元素 (段落和表格)
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            # --- 处理段落 ---
            # 检查段落中是否包含图片
            if contains_image(block):
                # 提取图片
                image_filename = f"image_{image_counter}.png"
                image_path = os.path.join(image_folder, image_filename)
                
                # 从段落中获取图片的 rId
                r_id = get_image_r_id(block)
                if r_id:
                    image_part = doc.part.related_parts[r_id]
                    with open(image_path, 'wb') as f:
                        f.write(image_part.blob)
                    
                    # 在 Markdown 中插入图片引用
                    relative_image_path = os.path.join('images', image_filename).replace('\\', '/')
                    markdown_lines.append(f"![{image_filename}]({relative_image_path})")
                    image_counter += 1
            else:
                # 处理标题和普通文本
                # 这里可以根据 style 进一步扩展，例如处理列表等
                if block.style.name.startswith('Heading 1'):
                    markdown_lines.append(f"# {block.text}")
                elif block.style.name.startswith('Heading 2'):
                    markdown_lines.append(f"## {block.text}")
                elif block.style.name.startswith('Heading 3'):
                    markdown_lines.append(f"### {block.text}")
                elif block.text.strip(): # 忽略空段落
                    markdown_lines.append(block.text)

        elif isinstance(block, Table):
            # --- 处理表格 ---
            # 将表格转换为 HTML
            html_table = "<table>\n"
            for row in block.rows:
                html_table += "  <tr>\n"
                for cell in row.cells:
                    # 使用 <td> 标签，也可以根据第一行判断使用 <th>
                    html_table += f"    <td>{cell.text}</td>\n"
                html_table += "  </tr>\n"
            html_table += "</table>"
            markdown_lines.append(html_table)
            
    # 将所有行合并成一个 Markdown 文本
    output_md_path = os.path.join(output_folder, os.path.basename(docx_path).replace('.docx', '.md'))
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(markdown_lines))
        
    print(f"转换完成！Markdown 文件保存在: {output_md_path}")
    return {'text_path': str(output_md_path),
            "image_path": str(image_folder),
            "temp_dir_path":str(output_folder)}


# --- 辅助函数 ---

def iter_block_items(parent):
    """
    生成器，用于按顺序迭代文档中的段落和表格。
    `parent` 可以是 Document 对象或者 Table Cell 对象。
    """
    if isinstance(parent, Document):
        parent_elm = parent.element.body
    else:
        raise ValueError("Unsupported parent type")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def contains_image(paragraph):
    """检查段落中是否包含 <w:drawing> 元素 (图片)"""
    return '<w:drawing>' in paragraph._p.xml

def get_image_r_id(paragraph):
    """从段落的 XML 中提取图片的 Relationship ID (rId)"""
    xml_str = paragraph._p.xml
    try:
        # 寻找 <a:blip> 标签中的 r:embed 属性
        embed_tag_start = xml_str.find('r:embed="')
        if embed_tag_start != -1:
            quote_start = embed_tag_start + len('r:embed="')
            quote_end = xml_str.find('"', quote_start)
            return xml_str[quote_start:quote_end]
    except Exception:
        return None
    return None


# --- 使用示例 ---
if __name__ == '__main__':
    # 假设您有一个名为 'mydocument.docx' 的文件
    # 它包含文本、标题、一张图片和一个表格
    try:
        res = docx_to_markdown('data/Documents/Document.docx', 'output')
        print(res)
    except Exception as e:
        print(f"创建或转换文档时出错: {e}")
        print("请确保你有一个名为 'test_document.docx' 的文件在脚本同目录下。")