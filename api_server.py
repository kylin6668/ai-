from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import shutil
import os
import uvicorn
from typing import Optional

# 复用现有项目逻辑（不修改原文件）
from qa_chain import VegetablePriceChatbot
from vector_db import init_vector_db


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_count: int
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    vectors: int
    model: str


class ApiServerState:
    _lock = threading.Lock()
    _bot: Optional[VegetablePriceChatbot] = None

    @classmethod
    def get_bot(cls) -> VegetablePriceChatbot:
        if cls._bot is None:
            with cls._lock:
                if cls._bot is None:
                    cls._bot = VegetablePriceChatbot()
        return cls._bot

    @classmethod
    def reset_bot(cls):
        with cls._lock:
            cls._bot = None


app = FastAPI(title="Cabbage Price QA API", version="1.0.0")

# CORS，可按需限制到你的前端主机
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    try:
        bot = ApiServerState.get_bot()
        try:
            vector_count = bot.db._collection.count()
        except Exception:
            vector_count = 0
        return HealthResponse(
            status="ok",
            vectors=vector_count,
            model="glm-z1-flash",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    bot = ApiServerState.get_bot()
    result = bot.chat(req.question)
    if isinstance(result, dict):
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return ChatResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            retrieved_count=int(result.get("retrieved_count", 0)),
        )
    # 字符串形式（如退出提示）
    return ChatResponse(answer=str(result), sources=[], retrieved_count=0)


@app.post("/rebuild")
def rebuild_vector_db():
    """
    重建向量库：删除现有持久化目录并重新创建，随后重置Bot以重新加载。
    注意：该操作会耗时，且会清空现有向量库。
    """
    try:
        persist_dir = "./chroma_db_zhipu"
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir)
        # 重新构建
        init_vector_db()
        # 重置与重载
        ApiServerState.reset_bot()
        ApiServerState.get_bot()
        return {"status": "ok", "message": "向量库已重建并重新加载"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 对外暴露到局域网
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)


