# qa_chain.py
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re

load_dotenv()


class VegetablePriceChatbot:
    def __init__(self):
        self.api_key = os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError("未找到ZHIPUAI_API_KEY，请在.env文件中配置")

        self.llm = ChatZhipuAI(
            api_key=self.api_key,
            model_name="glm-z1-flash",
            temperature=0.2,
            request_timeout=60
        )

        self.db, self.retriever = self._init_retriever()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chain, self.question_answer_chain = self._init_conversational_chain()
        print("白菜价格查询助手已启动，您可以开始提问了（输入'退出'结束对话）\n")

    # ---------------- 检索器初始化 ----------------
    def _init_retriever(self):
        embeddings = ZhipuAIEmbeddings(api_key=self.api_key, model="embedding-2")
        db = Chroma(persist_directory="./chroma_db_zhipu", embedding_function=embeddings)
        n = db._collection.count()
        if n == 0:
            raise ValueError("向量库为空，请先运行 vector_db.py")
        print(f"✅ 向量库加载成功，包含 {n} 条数据")
        return db, db.as_retriever(search_kwargs={"k": 6})

    # ---------------- 对话链初始化 ----------------
    def _init_conversational_chain(self):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "determine if the question can be answered using ONLY the chat history. "
            "If the question cannot be answered with the chat history alone "
            "(especially questions about vegetable prices, markets, or dates), "
            "reformulate it into a standalone question that can be used to search for information. "
            "If the question can be answered with the chat history, return it as is.\n"
            "Do NOT answer the question, only reformulate it if necessary."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        system_prompt = """
你是一个专业的蔬菜价格查询助手，必须严格按照以下规则回答：
1. 回答必须优先基于【上下文数据】（这是价格信息的唯一来源）
2. 仅当需要理解用户指代（如"它的价格"）时，才参考【历史对话】
3. 即使历史对话中有相关信息，也必须用最新的【上下文数据】验证

【历史对话】：
{chat_history}

【上下文数据】：
{context}

回答要求：
- 必须包含品种、市场、价格（最低价/最高价/平均价）和日期
- 如无相关数据，直接回复"未找到相关价格数据"
- 禁止编造任何信息，所有内容必须来自【上下文数据】
- 语言简洁自然，符合口语表达习惯
"""
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain), question_answer_chain

    # ---------------- 条件解析 ----------------
    def _parse_filters(self, question: str):
        q = str(question)

        # 日期
        date_full = None
        date_md = None
        if m := re.search(r"(\d{4})-(\d{2})-(\d{2})", q):
            date_full = m.group(0)
        if m := re.search(r"(\d{1,2})月(\d{1,2})日", q):
            date_md = f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"

        # 省份
        province = next((p for p in
                         ["北京", "天津", "上海", "重庆", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
                          "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东",
                          "广西", "海南", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"]
                         if p in q), None)

        # 城市（县/区/市）
        city = None
        if m := re.search(r"([\u4e00-\u9fa9]{2,3})(?:市|州|县|区)(?!\w)", q):
            city = m.group(1)

        # 具体市场
        market = None
        if m := re.search(r"([\u4e00-\u9fa9A-Za-z0-9·（）()\-]{4,}?)(市场|公司|批发市场|交易中心|有限公司)", q):
            market = m.group(0)

        # 品种
        variety = None
        for vk in ["大白菜", "白菜", "圆白菜", "洋白菜", "莲花白"]:
            if vk in q:
                variety = "大白菜" if vk != "大白菜" else vk
                break

        return {
            "date_full": date_full,
            "date_md": date_md,
            "province": province,
            "city": city,
            "market": market,
            "variety": variety
        }

    # ---------------- 构造 Chroma 合法 where ----------------
    def _build_where(self, filters: dict):
        clauses = []
        # 1. 市场 > 省份 > 城市  优先级
        if filters.get("market"):
            clauses.append({"market": {"$eq": filters["market"]}})
        else:
            if filters.get("province"):
                clauses.append({"province": {"$eq": filters["province"]}})
            if filters.get("city"):
                clauses.append({"city": {"$eq": filters["city"]}})

        # 日期
        if filters.get("date_full"):
            clauses.append({"date": {"$eq": filters["date_full"]}})
        elif filters.get("date_md"):
            clauses.append({"date_md": {"$eq": filters["date_md"]}})

        # 品种
        if filters.get("variety"):
            clauses.append({"variety": {"$eq": filters["variety"]}})

        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    # ---------------- 对话入口 ----------------
    def chat(self, question):
        if question.lower() in ["退出", "exit", "quit"]:
            return "对话结束，感谢使用！"
        try:
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            filters = self._parse_filters(question)
            where = self._build_where(filters)
            docs = []
            if where:
                docs = self.db.similarity_search(query="白菜 价格", k=10, filter=where)
            if docs:
                result = self.question_answer_chain.invoke({
                    "input": question,
                    "chat_history": chat_history,
                    "context": docs
                })
                answer = result["answer"] if isinstance(result, dict) and "answer" in result else result
                self.memory.save_context({"input": question}, {"output": answer})
                return {"answer": answer, "sources": [d.page_content for d in docs], "retrieved_count": len(docs)}
            # 兜底
            result = self.chain.invoke({"input": question, "chat_history": chat_history})
            self.memory.save_context({"input": question}, {"output": result["answer"]})
            return {"answer": result["answer"], "sources": [doc.page_content for doc in result["context"]],
                    "retrieved_count": len(result["context"])}
        except Exception as e:
            return {"error": str(e)}


# -------------------- 启动入口 --------------------
def main():
    try:
        bot = VegetablePriceChatbot()
        while True:
            question = input("您的问题: ")
            response = bot.chat(question)
            if "error" in response:
                print(f"错误: {response['error']}")
            else:
                print(f"\n回答: {response['answer']}")
                print(f"检索到 {response['retrieved_count']} 条相关数据")
                print("\n参考数据:")
                for i, src in enumerate(response["sources"], 1):
                    print(f"{i}. {src}")
                print("-" * 50)
    except Exception as e:
        print(f"程序启动失败: {str(e)}")


if __name__ == "__main__":
    main()