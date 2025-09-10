import pandas as pd
from langchain_text_splitters import CharacterTextSplitter
import re

# 加载数据
def load_cabbage_data(file_path):
    df = pd.read_csv(file_path)
    # 处理部分列存在的空值情况，用“无数据”填充
    df = df.fillna("无数据")
    # 将每行数据转换为文本格式
    df['text'] = df.apply(lambda row: f"品种：{row['品种']}，批发市场：{row['批发市场']}，最低价：{row['最低价']}元，最高价：{row['最高价']}元，平均价：{row['平均价']}元，发布日期：{row['发布日期']}", axis=1)
    return df['text'].tolist()

# 构建文本与元数据（用于精确过滤）
def build_texts_and_metadatas(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna("无数据")

    def extract_province(market: str) -> str:
        if not isinstance(market, str) or len(market) < 2:
            return "未知"
        # 简化处理：取前两个汉字作为省级区域（如“甘肃”、“北京”、“山东”）
        return market[:2]

    def extract_city(market: str) -> str:
        if not isinstance(market, str):
            return "未知"
        # 匹配“XX省/自治区/市?XX市/州/县/区”中的市县州等
        m = re.search(r"[省市自治区特别行政区]{0,3}([\u4e00-\u9fa5]{2,3})(市|州|县|区)", market)
        if m:
            return m.group(1)
        # 兜底：查找“市”前两个字
        m2 = re.search(r"([\u4e00-\u9fa5]{2,3})市", market)
        if m2:
            return m2.group(1)
        return "未知"

    def split_date(date_str: str):
        s = str(date_str)
        # 期望 YYYY-MM-DD，提取年、月、日
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
        if not m:
            return {"date": s, "year": "未知", "date_md": s}
        year, month, day = m.group(1), m.group(2), m.group(3)
        return {"date": s, "year": year, "date_md": f"{month}-{day}"}

    texts = []
    metadatas = []
    for _, row in df.iterrows():
        text = f"品种：{row['品种']}，批发市场：{row['批发市场']}，最低价：{row['最低价']}元，最高价：{row['最高价']}元，平均价：{row['平均价']}元，发布日期：{row['发布日期']}"
        texts.append(text)
        date_parts = split_date(row.get('发布日期', '未知'))
        metadatas.append({
            "variety": str(row.get('品种', '未知')),
            "market": str(row.get('批发市场', '未知')),
            "avg_price": str(row.get('平均价', '未知')),
            "min_price": str(row.get('最低价', '未知')),
            "max_price": str(row.get('最高价', '未知')),
            "date": date_parts["date"],
            "year": date_parts["year"],
            "date_md": date_parts["date_md"],
            "province": extract_province(str(row.get('批发市场', '未知'))),
            "city": extract_city(str(row.get('批发市场', '未知')))
        })
    return texts, metadatas

# 分割文本
def split_texts(texts):
    text_splitter = CharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separator=","
    )
    # 新版 API：split_text
    return text_splitter.split_text(texts)

if __name__ == "__main__":
    texts = load_cabbage_data("cabbage_prices.csv")
    split_texts = split_texts(texts)
    print(f"处理后的数据片段数量：{len(split_texts)}")