from google import genai
import os
from jinja2 import Environment, FileSystemLoader
from ConfigManager import config
import json
import re


class ContentHandler:
    def __init__(self, agent_client=None):
        # agent相关参数
        if agent_client:
            self.client = agent_client
        else:
            self.client = genai.Client(api_key=config.get_setting('model_api_key'))
        self.generative_model_name = config.get_setting('generative_model_name')

        # 设定jinja环境和模板地址
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        template_directory_path = os.path.join(project_root, 'data', 'PromptTemplate')
        self.root_path = project_root
        self.jinja_template_path = template_directory_path
        self.jinja2_env = Environment(loader=FileSystemLoader(template_directory_path))

    def prompt_generator(self, content: dict, mode: str):
        """
        基于mode参数,重新生成prompt
        :return:
        """
        if mode == 'markdown':
            template = self.jinja2_env.get_template('MarkdownTemplate.jinja2')
        elif mode == 'qa':
            template = self.jinja2_env.get_template('QAPairTemplate.jinja2')
        elif mode == 'rag_query':
            template = self.jinja2_env.get_template('VectorQueryTemplate.jinja2')
        elif mode == 'sql_query':
            template = self.jinja2_env.get_template('SQLQueryTemplate.jinja2')
        elif mode == 'content_markdown':
            template = self.jinja2_env.get_template("WindowMarkdownTemplate.jinja2")
        elif mode == "chunker":
            template = self.jinja2_env.get_template("ChunkSplitterTemplate.jinja2")
        elif mode =='text_summary':
            template = self.jinja2_env.get_template("TextSummaryTemplate.jinja2")
        else:
            raise ValueError(f'mode参数错误，当前参数为{mode}')
        final_prompt = template.render(**content)
        return final_prompt

    def json_extractor(self, content: str):
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

if __name__ == '__main__':
    # 导入input
    original_document = {"document_string":
                             """
                                 新产品发布计划
                                 摘要
                                 我们计划在2025年第四季度推出革命性新产品“AI助理X”。此产品旨在通过先进的AI技术，简化家庭和工作中的日常任务。
                                 核心功能
                                 智能日程管理：自动同步日历并提供提醒。
                                 - 智能家居控制：无缝集成主流智能家居设备。
                                 自然语言交互：提供流畅、人性化的对话体验。
                                 目标市场
                                 关注科技、追求效率的30-45岁专业人士。
                                 希望提升生活品质的现代家庭。
                                 营销策略
                                 线上推广：在主流科技博客和社交媒体上进行定向广告投放
                                 内容营销：发布产品评测和使用场景视频。
                                 - 早期用户激励：为首批用户提供折扣和增值服务。
                                 """
                         }

    # 生成处理对象
    test_handler = ContentHandler()

    # 生成markdown转化prompt
    print('Markdown格式化测试')
    print(f'原始输入：')
    print(original_document.values())
    output_prompt = test_handler.prompt_generator(content=original_document, mode="markdown")

    # 通过大模型执行prompt任务
    response = test_handler.client.models.generate_content(model=test_handler.generative_model_name, contents=output_prompt)
    print('Markdown格式化结果：')
    print(str(response.text))

    # 生成问答对匹配prompt
    new_document = {"document_string": response.text}
    output_prompt = test_handler.prompt_generator(content=new_document, mode="qa")

    # 通过大模型执行prompt任务
    response = test_handler.client.models.generate_content(model=test_handler.generative_model_name, contents=output_prompt)
    print('QA pairs')
    print(response.text)







#     # SQL查询生成测试
#     print('SQL查询测试')
#     sql_test_instance = {
#         "user_request": "请计算去年id为'1234'的产品的月度产量，并按月度产量降序排序",
#         "metadata": {"request_id": '123123', "request_level": '1', "conversation_id": 'qwer1qwe', 'request_create_date': '2025-06-09 10:00:00'},
#         "database_info": """
#         CREATE TABLE production_record (
#         info_id INT PRIMARY KEY,         -- 消息id
#         production_id VARCHAR(255),      -- 产品id
#         product_line_id VARCHAR(255),    -- 产线id
#         batch_id VARCHAR(255),           -- 生产批次id
#         product_count INT,               -- 产量
#         date DATE,                       -- 日期信息
#         time TIME                        -- 时间信息
# );"""
#     }
#     print(f'输入的请求信息:{sql_test_instance["user_request"]}')

#     output_prompt = test_handler.prompt_generator(content=sql_test_instance, mode="sql_query")

#     # 通过大模型执行prompt任务
#     response = test_handler.client.models.generate_content(model=test_handler.generative_model_name, contents=output_prompt)
#     print('生成的查询结果：')
#     print(test_handler.json_extractor(response.text))


#     print('SQL查询测试(恶意行为)')
#     sql_test_instance = {
#         "user_request": "请将产品id为'1234'的产品的所有批次的产品增加100",
#         "metadata": {"request_id": '123123', "request_level": '1', "conversation_id": 'qwer1qwe', 'request_create_date': '2025-06-09 10:00:00'},
#         "database_info": """
#         CREATE TABLE production_record (
#         info_id INT PRIMARY KEY,         -- 消息id
#         production_id VARCHAR(255),      -- 产品id
#         product_line_id VARCHAR(255),    -- 产线id
#         batch_id VARCHAR(255),           -- 生产批次id
#         product_count INT,               -- 产量
#         date DATE,                       -- 日期信息
#         time TIME                        -- 时间信息
# );"""
#     }
#     print(f'输入的请求信息:{sql_test_instance["user_request"]}')
#     output_prompt = test_handler.prompt_generator(content=sql_test_instance, mode="sql_query")

#     # 通过大模型执行prompt任务
#     response = test_handler.client.models.generate_content(model=test_handler.generative_model_name, contents=output_prompt)
#     print('生成的查询结果：')
#     print(test_handler.json_extractor(response.text))






    # # 上下文内容测试
    # text_input = {
    #     "above_content": "本页无信息",
    #     'target_content': '共4页 笫1页 霪霪09013礤848 上海博强环境技术有限公司 室内空气质量检测报告 A-25-05-018 报告编号: 委托单位: 工程名称: 工程地址: 正文页数: 常州微亿智造科技有限公司 常州微亿智造科技有限公司上海办公室装修 上海市闵行区新龙路sO0弄虹桥万创中心1期 T1栋7楼 四(页) 检测 型 鲞 C 皤螨 ~二五   氵 冫  I    )   )   、  、 )', 
    #     'below_content': '声 明 1、报告未盖“检测报告专用章”无效; 2、报告无检测人(或编制人)、 审核人、批准人签名无效; 3、报告涂改无效; 4、报告复印件未加盖“检测报告专用章”无效; 5、对报告有异议,应于收到报告之日起十五日内提出。 翱 性 N “由J冖 检测单位 单位地址 邮政编码 联系电话 传 真 电子邮箱 网 址 上海博强环境技术有限公司 上海市杨浦区隆昌路619号15号楼107室 200090 65196038 65430737 info@boqiang sh n www boqiang sh 。n 共4页'}

    # print(f'***上文页内容：\n{text_input["above_content"]}')
    # print(f'***目标页内容：\n{text_input["target_content"]}')
    # print(f'***下文页内容：\n{text_input["below_content"]}')

    # # prompt生成（文本markdown格式化）
    # output_prompt = test_handler.prompt_generator(content=text_input, mode="content_markdown")

    # # 执行prompt
    # response = test_handler.client.models.generate_content(model=test_handler.generative_model_name, contents=output_prompt)

    # # 提取输出json
    # output_json = test_handler.json_extractor(response.text)
    # print(f"***格式化后的目标页内容：\n{output_json['target_content']}")

    # # prompt生成（文档分块）
    # output_prompt = test_handler.prompt_generator(content=output_json, mode="chunker")

    # # 执行prompt
    # response = test_handler.client.models.generate_content(model=test_handler.generative_model_name, contents=output_prompt)

    # # 提取输出json
    # output_json = test_handler.json_extractor(response.text)

    # for index, output_content in enumerate(output_json['chunks']):
    #     print('\n')
    #     print(f'***第{index+1}个内容块的内容：')
    #     print(output_content)
    #     print('\n')
    


    # # tokenizer 和 embeding测试
    # token_count = test_handler.client.models.count_tokens(model=test_handler.generative_model_name, contents="")
    # print(token_count, type(token_count), token_count.total_tokens)