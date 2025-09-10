from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
import os
import time
import requests
from dotenv import load_dotenv
from data_processor import build_texts_and_metadatas

# 加载环境变量
load_dotenv()

# 初始化向量存储
def init_vector_db():
    try:
        # 加载文本与元数据（不拆分，保证逐行对齐，便于后续元数据过滤）
        texts, metadatas = build_texts_and_metadatas("cabbage_prices.csv")

        # 验证数据加载是否成功
        if not texts or not metadatas or len(texts) != len(metadatas):
            raise ValueError("未加载到有效数据，请检查cabbage_prices.csv文件")
        
        total = len(texts)
        print(f"共需处理 {total} 条数据，将分批次导入向量库...")
        
        # 获取API密钥
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("未找到ZHIPUAI_API_KEY，请在.env文件中配置")

        # 检查智谱API连接状态
        def check_zhipu_connection():
            try:
                response = requests.head(
                    "https://open.bigmodel.cn/api/paas/v4/",
                    timeout=10
                )
                return True
            except requests.exceptions.RequestException as e:
                print(f"智谱API连接测试失败: {str(e)}")
                return False

        if not check_zhipu_connection():
            raise ConnectionError("无法连接到智谱清言API，请检查网络连接")

        # 初始化智谱嵌入模型
        embeddings = ZhipuAIEmbeddings(
            api_key=api_key,
            model="embedding-2"  # 智谱官方推荐的嵌入模型
        )
        
        # 分批次处理（智谱API限制单次最多64条）
        batch_size = 60  # 留一点余量，避免触发限制
        total_batches = (total + batch_size - 1) // batch_size
        db = None
        
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total)
            batch_texts = texts[start_idx:end_idx]
            batch_metas = metadatas[start_idx:end_idx]
            
            print(f"正在处理第 {i+1}/{total_batches} 批次，包含 {len(batch_texts)} 条数据...")
            
            # 第一批创建数据库，后续批次添加数据
            if i == 0:
                db = Chroma.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metas,
                    persist_directory="./chroma_db_zhipu"
                )
            else:
                db.add_texts(texts=batch_texts, metadatas=batch_metas)
            
            db.persist()
            # 每批处理后休息1秒，避免请求过于频繁
            if i < total_batches - 1:
                time.sleep(1)
        
        print(f"全部数据处理完成，共导入 {total} 条白菜价格数据")
        return db

    except Exception as e:
        print(f"初始化向量数据库时出错: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        db = init_vector_db()
        print("向量数据库创建完成，可用于白菜价格问答系统")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
