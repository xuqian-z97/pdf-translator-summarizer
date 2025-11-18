# main.py
import os
import fitz  # PyMuPDF
from dashscope import Generation
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 配置CORS（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 从环境变量获取API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY not set in environment variables")


@app.post("/translate-and-summarize")
async def translate_and_summarize(file: UploadFile = File(...)):
    """
    接收PDF文件，返回翻译后的文本和摘要
    """
    # 检查文件类型
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # 读取PDF文件
    pdf_bytes = await file.read()

    # 提取文本
    text = extract_text_from_pdf(pdf_bytes)

    # 翻译并总结
    result = translate_and_summarize_text(text)

    return {
        "translated_text": result["translated"],
        "summary": result["summary"]
    }


def extract_text_from_pdf(pdf_bytes):
    """使用PyMuPDF提取PDF文本"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def translate_and_summarize_text(text):
    """调用DashScope API进行翻译和总结"""
    # 分块处理（避免超过模型上下文限制）
    chunks = split_text(text, 4000)

    # 翻译每个块
    translated_chunks = []
    for chunk in chunks:
        translated = translate_text(chunk)
        translated_chunks.append(translated)

    full_translated = "\n".join(translated_chunks)

    # 生成摘要
    summary = generate_summary(full_translated)

    return {
        "translated": full_translated,
        "summary": summary
    }


def split_text(text, chunk_size=4000):
    """将长文本分割成小块"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def translate_text(text):
    """翻译文本"""
    prompt = f"请将以下内容翻译成英文：\n\n{text}"
    response = Generation.call(
        model="qwen-max",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY
    )
    return response.output.text


def generate_summary(text):
    """生成文本摘要"""
    prompt = f"请对以下内容进行简洁总结，不超过200字：\n\n{text}"
    response = Generation.call(
        model="qwen-max",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY
    )
    return response.output.text


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)