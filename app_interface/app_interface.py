import gradio as gr
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from threading import Thread
import base64
import trimesh
import numpy as np
import tempfile
from trimesh.exchange.gltf import export_glb
import torch
import datetime


# 将图片转换为Base64字符串
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# @spaces.GPU(duration=120)
# 修改后的聊天生成函数（兼容Transformers API）
def generate_3d_mesh(user_prompt: str,
                   chat_history: list,
                   temperature_value: float,
                   top_p_value: float,
                   max_tokens: int) -> str:
    # 构建消息格式
    formatted_messages = []
    system_message = {"role": "system", "content": ""}
    formatted_messages.append(system_message)

    for user_message, assistant_response in chat_history:
        formatted_messages.append({"role": "user", "content": user_message})
        formatted_messages.append({"role": "assistant", "content": assistant_response})
    formatted_messages.append({"role": "user", "content": user_prompt})

    # 使用tokenizer.apply_chat_template构建提示
    model_prompt = language_tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 转换为模型输入
    tokenized_input = language_tokenizer(model_prompt, return_tensors="pt").to(language_model.device)

    # 添加设备同步
    if language_model.device.type == "meta":
        language_model.to("cuda" if torch.cuda.is_available() else "cpu")

    # 流式生成配置
    from transformers import TextIteratorStreamer
    text_streamer = TextIteratorStreamer(
        language_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # 在单独线程中生成
    generation_config = dict(
        tokenized_input,
        streamer=text_streamer,
        max_new_tokens=max_tokens,
        temperature=temperature_value,
        top_p=top_p_value,
        do_sample=True,
        eos_token_id=eos_tokens
    )
    generation_thread = Thread(target=language_model.generate, kwargs=generation_config)
    generation_thread.start()

    # 生成完成后处理输出
    complete_output = ""
    for generated_text in text_streamer:
        complete_output += generated_text

        # 实时过滤保留v/f行
        output_lines = complete_output.split('\n')
        if output_lines and output_lines[0].strip().startswith("Here"):
            mesh_lines = [
                line for line in output_lines
                if line.startswith(('v ', 'f '))
            ]
            filtered_mesh = '\n'.join(mesh_lines)
        else:
            # 如果不是以"Here"开头，则不过滤
            filtered_mesh = complete_output

        yield filtered_mesh

    # === 最终过滤处理 ===
    final_output_lines = complete_output.split('\n')
    # 仅当以"Here"开头时才过滤
    if final_output_lines and final_output_lines[0].strip().startswith("Here"):
        final_mesh_lines = [
            line.strip() for line in final_output_lines
            if line.strip().startswith(('v ', 'f '))
        ]
        final_mesh_data = '\n'.join(final_mesh_lines)
    else:
        final_mesh_data = complete_output

    # 保存过滤后的记录
    mesh_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": user_prompt,
        "mesh_data": final_mesh_data,  # 使用过滤后的数据
        "parameters": {
            "temperature": temperature_value,
            "top_p": top_p_value,
            "max_new_tokens": max_tokens
        }
    }
    global generation_history
    generation_history.append(mesh_record)

    # 在生成完成后释放GPU内存
    torch.cuda.empty_cache()

def apply_gray_wireframe(mesh_text):
    """
    生成灰色线框可视化
    Args:
        mesh_text (str): OBJ格式的网格文本
    Returns:
        str: GLB文件路径
    """
    temp_obj_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    with open(temp_obj_path, "w") as obj_file:
        obj_file.write(mesh_text)

    # 加载网格
    mesh_object = trimesh.load_mesh(temp_obj_path)

    # === 线框生成 ===
    mesh_edges = mesh_object.edges_unique
    wireframe_indices = np.repeat(mesh_edges, 2, axis=0).reshape(-1, 2)

    # 创建线段实体
    wireframe_lines = [
        trimesh.path.entities.Line(points=indices, color=(30, 30, 30, 255))
        for indices in wireframe_indices
    ]
    wireframe_object = trimesh.path.Path3D(
        entities=wireframe_lines,
        vertices=mesh_object.vertices.copy(),
        process=False
    )

    # === 颜色设置 ===
    mesh_object.visual.face_colors = (200, 200, 200, 255)  # 面片颜色

    # === 组合场景 ===
    mesh_scene = trimesh.Scene()
    mesh_scene.add_geometry(mesh_object)  # 面片
    mesh_scene.add_geometry(wireframe_object)  # 线框

    # === 生成预览用GLB ===
    preview_glb_path = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    mesh_scene.export(preview_glb_path)  # 保持预览功能

    # === 导出标准OBJ ===
    export_obj_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    with open(export_obj_path, 'w') as obj_file:
        # 写入顶点（整数坐标）
        for vertex in mesh_object.vertices:
            x_coord = int(round(vertex[0]))
            y_coord = int(round(vertex[1]))
            z_coord = int(round(vertex[2]))
            obj_file.write(f"v {x_coord} {y_coord} {z_coord}\n")  # 显式换行

        obj_file.write("\n")  # 顶点与面片间空行

        # 写入面片
        for face in mesh_object.faces:
            # OBJ面片索引从1开始
            vertex1 = face[0] + 1
            vertex2 = face[1] + 1
            vertex3 = face[2] + 1
            obj_file.write(f"f {vertex1} {vertex2} {vertex3}\n")

    return preview_glb_path  # 返回GLB路径用于预览

# OBJ导出函数
def export_obj_file(mesh_text):
    """ 纯OBJ导出（无颜色） """
    # 二次过滤确保输入纯净
    filtered_lines = [
        line for line in mesh_text.split('\n')
        if line.startswith(('v ', 'f '))
    ]
    clean_mesh_text = '\n'.join(filtered_lines)

    temp_obj_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    with open(temp_obj_path, "w") as obj_file:
        obj_file.write(clean_mesh_text)  # 写入过滤后的内容

    mesh_object = trimesh.load_mesh(temp_obj_path)

    final_obj_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    with open(final_obj_path, 'w') as obj_file:
        # 显式循环写入顶点
        for vertex in mesh_object.vertices:
            obj_file.write("v {} {} {}\n".format(
                int(round(vertex[0])),
                int(round(vertex[1])),
                int(round(vertex[2]))
            ))

        obj_file.write("\n")  # 确保空行分隔

        # 显式循环写入面片
        for face in mesh_object.faces:
            obj_file.write("f {} {} {}\n".format(
                face[0] + 1,
                face[1] + 1,
                face[2] + 1
            ))

    return final_obj_path

def visualize_mesh(mesh_text):
    """
    Convert the provided 3D mesh text into a visualizable format.
    This function assumes the input is in OBJ format.
    """
    temp_mesh_file = "temp_mesh.obj"
    with open(temp_mesh_file, "w") as obj_file:
        obj_file.write(mesh_text)
    return temp_mesh_file

# 刷新历史记录列表
def update_history_list():
    global generation_history
    history_options = [
        f"{record['timestamp']} - {record['description'][:30]}..."
        for record in generation_history
    ]
    return gr.Dropdown(choices=history_options)

# 显示记录详情
def display_history_details(index):
    if index is not None and 0 <= index < len(generation_history):
        return generation_history[index]
    return {}

# 加载选中记录
def load_history_record(index):
    if index is not None and 0 <= index < len(generation_history):
        return generation_history[index]["mesh_data"]
    return ""


# 获取Logo图片的Base64编码
logo_base64 = convert_image_to_base64("/home/featurize/work/XDMesh/OSK1.png")

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# 界面顶部描述区域的HTML
UI_HEADER = f'''
<div style="
    background: linear-gradient(145deg, #FFFCF0, #F0FFFC);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 8px 8px 16px rgba(210, 226, 222, 0.7), 
               -8px -8px 16px #ffffff;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    border: 1px solid #2D8A78;
">
    <img src="data:image/jpeg;base64,{logo_base64}" 
         style="
            position: absolute;
            width: 120px;
            height: 120px;
            left: 30px;
            top: 50%;
            transform: translateY(-50%);
            border-radius: 15px;
            box-shadow: 4px 4px 12px rgba(0,0,0,0.1);
            border: 2px solid #2D8A78;
            object-fit: cover;
            z-index: 2;
         ">
    <!-- 其他元素保持不变 -->
    <div style="
        position: absolute;
        width: 300px;
        height: 300px;
        background: linear-gradient(45deg, #4AB19A55, #F8D48A55);
        right: -50px;
        top: -50px;
        border-radius: 50%;
        filter: blur(80px);
    "></div>
    <h1 style="
        color: #2D665A;
        font-size: 2.8em;
        text-align: center;
        margin-bottom: 20px;
        font-family: 'Arial Rounded MT Bold';
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        position: relative;
        margin-left: 140px;
    ">
        <span style="color: #4AB19A">⚡</span> XDMesh 3D 
        <span style="color: #F8D48A">Δ</span>
    </h1>
    <p style="
        color: #2D665A;
        font-size: 1.2em;
        text-align: center;
        line-height: 1.6;
        font-weight: 500;
        margin-left: 140px;
    ">
        基于大语言模型的3D网格生成
    </p>
</div>
'''

# 页脚许可证信息
UI_FOOTER = """
<div style="
    margin-top: 20px;
    padding: 15px;
    background: rgba(210, 226, 222, 0.8);
    border-radius: 20px;
    text-align: center;
    color: #2D665A;
    border: 1px solid #2D8A78;
    backdrop-filter: blur(5px);
    box-shadow: 4px 4px 12px rgba(0,0,0,0.05);
">
    <span style="font-family: 'Courier New'">>>_ </span> Powered by XDMesh
</div>
"""

# 输入框占位符提示
INPUT_PLACEHOLDER = """
<div style="
    padding: 30px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    opacity: 0.6;
    color: #2D665A;
">
   <h1 style="font-size: 28px; margin-bottom: 10px; color: #2D665A;">输入语言描述以生成3D网格</h1>
   <p style="font-size: 16px; color: #2D665A;">等待输入</p>
</div>
"""

# UI样式表
ui_css = """
/* 全局色调统一 */
:root {
    --main-bg: linear-gradient(145deg, #FFFCF0, #F0FFFC);
    --panel-bg: rgba(255, 255, 250, 0.9);
    --accent-color: #4AB19A;
    --secondary-color: #F8D48A;
    --text-color: #2D665A;
    --text-secondary: #936D31;
    --border-color: #2D8A78; /* 深青绿色边框颜色 */
    --shadow-light: 8px 8px 16px rgba(210, 226, 222, 0.7), -8px -8px 16px rgba(255, 255, 255, 0.8);
    --shadow-inset: inset 4px 4px 8px rgba(210, 226, 222, 0.5), inset -4px -4px 8px rgba(255, 255, 255, 0.7);
    --border-radius: 20px;
    --font-size-base: 16px;
    --font-size-heading: 1.2em;
    --font-family: 'Segoe UI', system-ui, sans-serif;
    --gradient-accent: linear-gradient(145deg, var(--accent-color), #3A9080);
    --gradient-secondary: linear-gradient(145deg, var(--secondary-color), #E6C070);
}

/* 全局容器样式 */
.gradio-container {
    background: var(--main-bg) !important;
    font-family: var(--font-family);
    font-size: var(--font-size-base);
    color: var(--text-color);
}

/* 统一面板样式 */
.params-panel, .model3d-container, .chatbot, .textbox, .gr-accordion, 
.guide-card, .compact-download, .gr-dropdown, .gr-slider {
    background: var(--panel-bg) !important;
    border-radius: var(--border-radius) !important;
    box-shadow: var(--shadow-light) !important;
    border: 1px solid var(--border-color) !important; /* 添加深青绿色边框 */
    backdrop-filter: blur(8px);
}

/* 输入框统一样式 */
.gr-textbox, .gr-chatbot textarea, #mesh-input {
    background: var(--panel-bg) !important;
    border-radius: var(--border-radius) !important;
    box-shadow: var(--shadow-inset) !important;
    border: 1px solid var(--border-color) !important; /* 修改为深青绿色边框 */
    padding: 20px !important;
    transition: all 0.3s ease !important;
    color: var(--text-color);
    font-size: var(--font-size-base);
}

/* 聚焦状态 */
.gr-textbox:focus, .gr-chatbot textarea:focus {
    box-shadow: var(--shadow-inset) !important;
    border: 2px solid var(--border-color) !important; /* 加粗聚焦状态边框 */
}

/* 全局选项卡样式改造 */
.gr-tabs {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

.gr-tabs-list {
    gap: 12px !important;
    margin-bottom: 25px !important;
}

.gr-tab-item {
    background: var(--panel-bg) !important;
    border-radius: var(--border-radius) !important;
    border: 1px solid var(--border-color) !important; /* 修改为深青绿色边框 */
    box-shadow: var(--shadow-light) !important;
    padding: 12px 24px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    color: var(--text-color) !important;
    font-size: var(--font-size-base);
}

.gr-tab-item:hover:not(.selected) {
    transform: translateY(-2px);
    box-shadow: 10px 10px 20px rgba(210, 226, 222, 0.8),
               -10px -10px 20px rgba(255, 255, 255, 0.9) !important;
}

.gr-tab-item.selected {
    background: linear-gradient(145deg, var(--accent-color), #3A9080) !important;
    color: white !important;
    box-shadow: 3px 3px 8px rgba(0,0,0,0.15),
               -2px -2px 4px rgba(255,255,255,0.5) !important;
}

/* 指南面板统一样式 */
.guide-panel {
    margin: 10px !important;
    background: transparent !important;
    border: none !important;
    height: 400px;
}

.guide-panel .gr-accordion-title {
    background: linear-gradient(145deg, var(--accent-color), #3A9080) !important;
    color: white !important;
    border-radius: var(--border-radius) var(--border-radius) 0 0;
    padding: 16px 24px !important;
    font-size: var(--font-size-heading);
}

.guide-card {
    background: var(--panel-bg);
    padding: 24px;
    border-radius: 0 0 var(--border-radius) var(--border-radius);
    box-shadow: var(--shadow-light);
    border: 1px solid var(--border-color) !important; /* 添加深青绿色边框 */
    border-top: none !important; /* 移除顶部边框，避免与标题重叠 */
    height: calc(100% - 60px);
    display: flex;
    flex-direction: column;
    justify-content: center; /* 垂直居中 */
    align-items: center; /* 水平居中 */
    text-align: center; /* 文本居中 */
}

.guide-card ol, .guide-card ul {
    color: var(--text-color);
    line-height: 1.8;
    padding-left: 20px;
    margin: 15px auto; /* 上下边距，左右自动居中 */
    font-size: var(--font-size-base);
    display: inline-block; /* 使列表项能够被居中 */
    text-align: left; /* 列表项文本左对齐，但整体居中 */
}

.guide-card code {
    background: rgba(74, 177, 154, 0.1);
    padding: 12px 16px;
    border-radius: var(--border-radius);
    display: block;
    margin: 15px auto; /* 上下边距，左右自动居中 */
    color: var(--text-color);
    font-size: var(--font-size-base);
    max-width: 80%; /* 限制宽度 */
}

.guide-card .examples {
    margin-top: 20px;
    border-top: 1px dashed var(--border-color);
    padding-top: 15px;
    width: 80%; /* 限制宽度 */
    margin-left: auto;
    margin-right: auto;
}

.guide-card.highlight {
    background: linear-gradient(145deg, #FFFCF0, #F0FFFC);
}

/* 历史记录页面样式 */
#history .gr-dropdown {
    background: var(--panel-bg) !important;
    border-radius: var(--border-radius) !important;
    border: 1px solid rgba(74, 177, 154, 0.3) !important;
    box-shadow: var(--shadow-inset) !important;
    padding: 14px 20px !important;
    color: var(--text-color);
    font-size: var(--font-size-base);
}

#history .gr-button {
    border-radius: var(--border-radius) !important;
    padding: 14px 24px !important;
    margin-left: 15px !important;
    font-size: var(--font-size-base);
}

#history .gr-json {
    background: var(--panel-bg) !important;
    border-radius: var(--border-radius) !important;
    padding: 24px !important;
    margin: 15px 0 !important;
    box-shadow: var(--shadow-light) !important;
    min-height: 400px !important;
    color: var(--text-color);
    font-size: var(--font-size-base);
}

/* 高度平衡处理 */
#preview .gr-row, #generation .gr-row {
    min-height: 600px !important;
    align-items: stretch !important;
}

#preview .gr-column, #generation .gr-column {
    height: 100% !important;
}

#preview .gr-model3d {
    height: 100% !important;
}

/* 预览下载页调整 */
.compact-download {
    max-width: 100% !important;
    margin: 15px 0 !important;
    padding: 15px !important;
    border-radius: var(--border-radius) !important;
    background: var(--panel-bg) !important;
    box-shadow: var(--shadow-light) !important;
    height: auto !important;
    max-height: 80px !important;
    overflow: hidden;
}

.compact-download .name {
    font-size: 1em !important;
    color: var(--text-color) !important;
}

.compact-download .size {
    font-size: 0.9em !important;
    color: var(--text-secondary) !important;
}

/* 统一按钮样式 */
.gr-button, #visualize-btn {
    background: linear-gradient(145deg, var(--accent-color), #3A9080) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--border-radius) !important;
    padding: 14px 24px !important;
    box-shadow: 4px 4px 10px rgba(210, 226, 222, 0.5),
               -4px -4px 10px rgba(255, 255, 255, 0.6) !important;
    transition: all 0.3s ease !important;
    font-size: var(--font-size-base);
}

.gr-button.secondary, .gr-button[variant="secondary"] {
    background: linear-gradient(145deg, var(--secondary-color), #E6C070) !important;
    color: var(--text-secondary) !important;
}

.gr-button:hover, #visualize-btn:hover {
    transform: translateY(-2px);
    box-shadow: 6px 6px 14px rgba(210, 226, 222, 0.6),
               -6px -6px 14px rgba(255, 255, 255, 0.7) !important;
}

/* 统一按钮组样式 */
.gr-button-group {
    justify-content: center !important;
    gap: 15px !important;
}

.gr-button-group .gr-button {
    flex: 1 !important;
    max-width: 200px !important;
    margin: 5px !important;
}

/* 滑块特效 */
.slider-container {
    padding: 20px 0 !important;
    background: rgba(245, 252, 249, 0.6) !important;
    border-radius: var(--border-radius) !important;
    margin: 15px 0 !important;
}

.slider-container .gr-slider {
    --thumb-size: 22px;
    --thumb-color: var(--accent-color);
    --track-color: rgba(74, 177, 154, 0.2);
    --track-height: 8px;
}

.slider-container .gr-number {
    font-weight: 600 !important;
    color: var(--text-color) !important;
    background: rgba(74, 177, 154, 0.1) !important;
    padding: 5px 10px !important;
    border-radius: 10px !important;
    font-size: var(--font-size-base);
}

/* 聊天气泡样式 */
.message.user {
    background: rgba(74, 177, 154, 0.1) !important;
    border: 1px solid var(--border-color) !important; /* 修改为深青绿色边框 */
    border-radius: var(--border-radius) !important;
    color: var(--text-color);
    font-size: var(--font-size-base);
}

.message.assistant {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-color) !important; /* 修改为深青绿色边框 */
    border-radius: var(--border-radius) !important;
    color: var(--text-color);
    font-size: var(--font-size-base);
}

/* 参数面板标题 */
.params-panel .gr-accordion-title {
    color: var(--text-color) !important;
    font-size: var(--font-size-heading) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* 参数面板标题装饰 */
.params-panel .gr-accordion-title::before {
    content: "⚙️";
    margin-right: 8px;
    filter: drop-shadow(2px 2px 2px rgba(0,0,0,0.1));
}

/* 示例样式 */
.gradio-examples {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px !important;
    padding: 24px !important;
}

.gradio-example {
    background: var(--panel-bg) !important;
    border-radius: var(--border-radius) !important;
    border: 1px solid var(--border-color) !important; /* 修改为深青绿色边框 */
    box-shadow: var(--shadow-light) !important;
    padding: 20px !important;
    text-align: center !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    color: var(--text-color);
    font-size: var(--font-size-base);
}

.gradio-example:hover {
    transform: translateY(-3px);
    box-shadow: 10px 10px 20px rgba(210, 226, 222, 0.8),
               -10px -10px 20px rgba(255, 255, 255, 0.9) !important;
}

/* 生成历史页面高度调整 */
#history .gr-column {
    min-height: 85vh !important; /* 增加到1.5倍高度 */
    display: flex !important;
    flex-direction: column !important;
}

#history .gr-row {
    margin-bottom: 15px !important;
}

/* 确保历史页面元素铺满屏幕 */
#history .gr-json {
    flex-grow: 1 !important;
    margin-bottom: 20px !important;
    min-height: 600px !important; /* 增加到1.5倍高度 */
}

#history .gr-dropdown {
    padding: 18px 24px !important; /* 增加内边距 */
    height: 60px !important; /* 增加高度 */
}

#history .gr-button {
    padding: 18px 30px !important; /* 增加按钮内边距 */
    height: 60px !important; /* 增加按钮高度 */
}

/* 网格生成页面高度对齐 */
#generation .gr-column {
    display: flex !important;
    flex-direction: column !important;
}

#generation .params-panel {
    flex-grow: 1 !important;
}

#generation .gr-accordion {
    height: 100% !important;
}

/* 统一过渡动画 */
.gr-component {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* 动态光效装饰 */
.gradio-container::before {
    content: '';
    position: fixed;
    width: 200vw;
    height: 200vh;
    background: radial-gradient(circle, rgba(74, 177, 154, 0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* 响应式布局优化 */
@media (max-width: 768px) {
    .gr-row {
        flex-direction: column !important;
    }

    .gr-column {
        width: 100% !important;
        margin-bottom: 20px !important;
    }

    #instruction .gr-markdown {
        margin-left: 0 !important;
    }
    
    :root {
        --font-size-base: 14px;
    }
}

/* 操作说明页面布局调整 */
#instruction .gr-row {
    min-height: 500px !important; /* 固定高度为500px */
    align-items: stretch !important;
}

#instruction .gr-column {
    height: 500px !important; /* 固定高度为500px */
    display: flex !important;
    flex-direction: column !important;
}

#instruction .gr-accordion {
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}

/* 预览下载页面输入框高度调整 */
#preview #mesh-input {
    height: 300px !important; /* 减小高度以匹配左侧预览区 */
    # resize: none !important; /* 禁止调整大小 */
}

/* 预览下载页面布局调整 */
#preview .gr-row {
    align-items: flex-start !important; /* 从顶部对齐 */
}

/* 修改LICENSE样式，使其符合新的色调 */
.gr-markdown:last-child {
    margin-top: 20px;
    padding: 15px;
    background: rgba(74, 177, 154, 0.1) !important;
    border-radius: var(--border-radius) !important;
    text-align: center;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    backdrop-filter: blur(5px);
    box-shadow: var(--shadow-light) !important;
}

/* 修改所有剩余的色调引用 */
.progress-bar {
    background: var(--gradient-accent) !important;
    border-radius: 8px !important;
    height: 8px !important;
}

.gr-form {
    border: 1px solid var(--border-color) !important;
    border-radius: var(--border-radius) !important;
    background: var(--panel-bg) !important;
}

.gr-input, .gr-select {
    border: 1px solid var(--border-color) !important;
    color: var(--text-color) !important;
}

.gr-check {
    border: 1px solid var(--border-color) !important;
    background: var(--panel-bg) !important;
}

.gr-check:checked {
    background: var(--accent-color) !important;
}

/* 确保示例中的代码块也符合色调 */
pre, code {
    background: rgba(74, 177, 154, 0.1) !important;
    border: 1px solid rgba(74, 177, 154, 0.2) !important;
    color: var(--text-color) !important;
}

/* 修正按钮样式 */
.gr-button-primary {
    background: var(--gradient-accent) !important;
}

.gr-button-secondary {
    background: var(--gradient-secondary) !important;
}

/* 确保进度条风格统一 */
.progress-track {
    background: rgba(74, 177, 154, 0.1) !important;
}

.progress-text {
    color: var(--text-color) !important;
}

/* 添加一个全局重置确保没有遗漏的紫色调 */
[style*="#9d4edd"], [style*="#7b1fa2"], [style*="#6a1b9a"], [style*="#7e57c2"] {
    color: var(--text-color) !important;
}

[style*="background: #9d4edd"], [style*="background: #7b1fa2"], 
[style*="background: #6a1b9a"], [style*="background: #7e57c2"] {
    background: var(--accent-color) !important;
}
"""

# 生成历史记录存储
generation_history = []

# 模型配置
model_directory = "./"  # 模型所在目录
model_filename = ".gguf"  # GGUF文件名
model_offload_directory = "./offload"  # 权重卸载目录

# 创建卸载目录
os.makedirs(model_offload_directory, exist_ok=True)

# 初始化分词器和模型
language_tokenizer = AutoTokenizer.from_pretrained(
    model_directory,
    gguf_file=model_filename
)

language_model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    gguf_file=model_filename,
    device_map="auto",  # 自动设备分配
    offload_folder=model_offload_directory,  # 指定卸载目录
    low_cpu_mem_usage=True,  # 内存优化模式
    torch_dtype=torch.float16,  # 半精度模式
    trust_remote_code=True,  # 信任远程代码
    offload_state_dict=True
)

# 定义模型终止符标记
eos_tokens = language_tokenizer.encode("<|eot_id|>", add_special_tokens=False)

with gr.Blocks(fill_height=True, css=ui_css) as app_interface:
    gr.Markdown(UI_HEADER)  # 顶部描述

    with gr.Tabs() as main_tabs:
        # ========== 操作说明页面 ==========
        with gr.Tab("📘 操作说明", id="instruction"):
            with gr.Blocks():
                with gr.Column():
                    with gr.Row():
                        # 第一列：生成流程
                        with gr.Column(scale=1, min_width=400):
                            with gr.Accordion("▌ 生成流程", open=True, elem_classes="guide-panel"):
                                gr.Markdown("""
                                <div class="guide-card", style="height: 400px;">
                                    <ol>
                                        <li>在【网格生成】页输入英文描述</li>
                                        <li>调整右侧参数控制生成效果</li>
                                        <li>点击生成按钮获取网格数据</li>
                                        <li>前往【预览下载】页进行可视化或导出</li>
                                    </ol>
                                </div>
                                """)
    
                        # 第二列：参数说明
                        with gr.Column(scale=1, min_width=300):
                            with gr.Accordion("▌ 参数说明", open=True, elem_classes="guide-panel"):
                                gr.Markdown("""
                                <div class="guide-card", style="height: 400px;">
                                    <ul>
                                        <li><strong>发散程度</strong>：创意性控制（0-1）</li>
                                        <li><strong>聚焦范围</strong>：词汇选择精度（0-1）</li>
                                        <li><strong>生成长度</strong>：建议4096左右</li>
                                    </ul>
                                </div>
                                """)
    
                        # 第三列：最佳实践
                        with gr.Column(scale=1, min_width=300):
                            with gr.Accordion("▌ 最佳实践", open=True, elem_classes="guide-panel"):
                                gr.Markdown("""
                                <div class="guide-card highlight", style="height: 400px;font-size: 12px">
                                    <p>✅ 推荐结构：</p>
                                    <code>"Create a 3D model of [物体]"</code>
                                    <div class="examples">
                                        <p>🌰 示例：</p>
                                        <ul>
                                            <li>Create a 3D model of a box</li>
                                            <li>Create a 3D model of a cube</li>
                                            <li>Create a 3D model of a chair</li>
                                        </ul>
                                    </div>
                                </div>
                                """)
                    gr.Markdown(UI_FOOTER)

        # ========== 网格生成页面 ==========
        with gr.Tab("🛠️ 网格生成", id="generation"):
            with gr.Column():
                #
                with gr.Row():
                    with gr.Column(scale=1):
                        # 参数面板保持不变
                        with gr.Accordion("⚙️ 调整参数", open=True, elem_classes="params-panel"):
                            with gr.Group():
                                temperature_slider = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    step=0.1,
                                    value=0.6,
                                    label="发散程度（Temperature）",
                                    info="控制回答的创意性："
                                         "\n🔥 调高（接近1）：回答更有想象力，但可能不准确"
                                         "\n🧊 调低（接近0）：回答更保守严谨，但可能较死板"
                                )
                                top_p_slider = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    step=0.01,
                                    value=0.9,
                                    label="聚焦范围（Top-p）",
                                    info="控制回答的专注度："
                                         "\n🎯 调高（接近1）：考虑更多可能性，回答更丰富"
                                         "\n🎯 调低（接近0）：只选最相关词汇，回答更集中"
                                )
                                max_tokens_slider = gr.Slider(
                                    minimum=128,
                                    maximum=8192,
                                    step=1,
                                    value=4096,
                                    label="回答长度（Max Tokens）",
                                    info="控制生成内容的长短："
                                         "\n📏 调大：生成更长更详细的回答"
                                         "\n📏 调小：生成更简短扼要的回答"
                                )

                    with gr.Column(scale=2):
                        # 聊天界面保持不变
                        gr.ChatInterface(
                            fn=generate_3d_mesh,
                            type="messages",
                            chatbot=gr.Chatbot(
                                height=450,
                                placeholder=INPUT_PLACEHOLDER,
                                label='3D网格生成对话窗口',
                                container=True,
                                # bubble_full_width=False,
                                type="messages",
                                elem_classes="chatbot",
                            ),
                            # 将手动创建的参数组件作为附加输入
                            additional_inputs=[
                                temperature_slider,
                                top_p_slider,
                                max_tokens_slider
                            ],
                            # examples=[
                            #     ['Create a 3D model of a box'],
                            #     ['Create a 3D model of a cube'],
                            #     [
                            #         'Create a 3D model of A matte white, geometric cube with a recessed square on one face and a diagonal indent across the corner of another face.'],
                            #     [
                            #         'Design a 3D object based on a yellow-top coffee table with black legs and a ceiling light fixture.'],
                            #     ['Generate a 3D obj file for a pink computer desk with a laptop on it.'],
                            #     ['Write a python code for sorting.'],
                            #     ['How to setup a human base on Mars? Give short answer.'],
                            #     ['What is 9,000 * 9,000?']
                            # ],
                            cache_examples=False,
                            submit_btn="🚀 生成",
                        )
                        gr.Markdown(UI_FOOTER)

        # ========== 预览下载页面 ==========
        with gr.Tab("🔍 预览下载", id="preview"):
            with gr.Row():
                with gr.Column():
                    mesh_preview = gr.Model3D(
                        label="3D网格预览",
                        interactive=False,
                        elem_classes="model3d-container",
                        height=500
                    )
                    gr.Markdown(UI_FOOTER)
    
                with gr.Column():
                    mesh_text_input = gr.Textbox(
                        label="输入点面数据",
                        placeholder="在这里粘贴OBJ文件",
                        lines=20,
                        elem_id="mesh-input",
                        container=False
                    )
    
                    with gr.Row():
                        preview_button = gr.Button(
                            "🌌 开始预览", variant="primary",
                            elem_id="visualize-btn", scale=1
                        )
                        download_obj_button = gr.Button(
                            "💾 导出OBJ", variant="secondary",
                            elem_id="visualize-btn", scale=1
                        )
    
                    # 按钮绑定保持不变
                    preview_button.click(
                        fn=apply_gray_wireframe,
                        inputs=[mesh_text_input],
                        outputs=[mesh_preview]
                    )
                    download_obj_button.click(
                        fn=export_obj_file,
                        inputs=[mesh_text_input],
                        outputs=gr.File(
                            label="下载OBJ文件",
                            elem_classes="compact-download"
                        )
                    )

        # ========== 生成历史页面 ==========
        with gr.Tab("📜 生成历史", id="history"):
            with gr.Column():
                with gr.Row():
                    history_selection = gr.Dropdown(
                        label="选择历史记录",
                        choices=[],
                        type="index",
                        interactive=True,
                        container=False,
                        scale=4
                    )
                    refresh_button = gr.Button(
                        "🔄 刷新",
                        variant="secondary",
                        scale=1,
                        min_width=100
                    )

                history_details = gr.JSON(
                    label="记录详情",
                    container=False,
                    elem_classes="params-panel"
                )
                load_history_button = gr.Button(
                    "📂 加载选中记录",
                    variant="primary",
                    scale=2,
                    min_width=200
                )
                gr.Button(
                    visible=False,
                    scale=1
                )

                refresh_button.click(
                    fn=update_history_list,
                    outputs=history_selection
                )
                history_selection.change(
                    fn=display_history_details,
                    inputs=history_selection,
                    outputs=history_details
                )
                load_history_button.click(
                    fn=load_history_record,
                    inputs=history_selection,
                    outputs=mesh_text_input
                )
                gr.Markdown(UI_FOOTER)

if __name__ == "__main__":
    app_interface.launch(share=True,
                server_name='0.0.0.0',
                server_port=7860,
                pwa=True)