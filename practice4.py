# version4.py
import json
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict
import aiofiles
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@dataclass
class QARecord:
    current_segment: str
    context: str
    target: str

async def load_json_async(filepath: str) -> List[QARecord]:
    async with aiofiles.open(filepath, mode='r', encoding='utf-8') as f:
        content = await f.read()
        data = json.loads(content)
        return [QARecord(**item) for item in data]

async def validate_async(record: QARecord, models, config) -> List[Tuple[str, str]]:
    anomalies = []

    if not record.current_segment:
        anomalies.append(("missing_value", "Пустой вопрос"))
    if not record.context:
        anomalies.append(("missing_value", "Пустой контекст"))
    if not record.target:
        anomalies.append(("missing_value", "Пустой ответ"))

    if len(record.context.split()) < 5:
        anomalies.append(("short_context", "Короткий контекст"))

    if len(record.target.split()) > 20:
        anomalies.append(("long_target", "Длинный ответ"))

    try:
        emb_ctx = models["embed"].encode(record.context)
        emb_tgt = models["embed"].encode(record.target)
        sim = cosine_similarity([emb_ctx], [emb_tgt])[0][0]

        if sim < config["semantic_threshold"]:
            anomalies.append(("low_semantic_similarity", f"Сходство: {sim:.2f}"))

        result = models["zero_shot"](
            record.context,
            candidate_labels=[record.target, "противоречие", "нерелевантно"]
        )
        if result['labels'][0] in ["противоречие", "нерелевантно"]:
            anomalies.append(("conceptual_inconsistency", f"Метка: {result['labels'][0]}"))

        contradiction = models["zero_shot"](
            record.context,
            candidate_labels=["подтверждение", "противоречие", "нейтрально"],
            hypothesis_template="Ответ: {}"
        )
        if contradiction['labels'][0] == "противоречие" and contradiction['scores'][0] > config["contradiction_threshold"]:
            anomalies.append(("contradictory_context", f"Противоречие: {contradiction['scores'][0]:.2f}"))

    except Exception as e:
        anomalies.append(("error", str(e)))

    return anomalies

async def validate_all(data: List[QARecord], models, config):
    results = []
    anomaly_counter = defaultdict(int)
    valid_count = 0

    for record in data:
        anomalies = await validate_async(record, models, config)
        if not anomalies:
            valid_count += 1
        else:
            for atype, _ in anomalies:
                anomaly_counter[atype] += 1

        results.append({
            "current_segment": record.current_segment,
            "context": record.context,
            "target": record.target,
            "valid": not anomalies,
            "anomalies": anomalies
        })

    return results, anomaly_counter, valid_count

async def save_json_log(results: List[Dict], filename: str = "validation_log.json"):
    async with aiofiles.open(filename, mode='w', encoding='utf-8') as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=2))

def visualize_anomalies(anomaly_counter: Dict[str, int], valid_count: int):
    sns.set(style="whitegrid")
    df = pd.DataFrame(list(anomaly_counter.items()), columns=["Тип аномалии", "Количество"])
    df = df.sort_values(by="Количество", ascending=False)

    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("coolwarm", len(df))
    sns.barplot(x="Тип аномалии", y="Количество", data=df, palette=palette)
    plt.title("Количество аномалий по типам")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


async def main_async(filepath: str):
    config = {
        "semantic_threshold": 0.5,
        "contradiction_threshold": 0.7
    }

    models = {
        "embed": SentenceTransformer('all-MiniLM-L6-v2'),
        "zero_shot": pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    }

    print("Загрузка данных...")
    data = await load_json_async(filepath)

    print("Валидация...")
    results, anomaly_counter, valid_count = await validate_all(data, models, config)

    print("Сохранение логов...")
    await save_json_log(results)

    print(f"Всего записей: {len(data)}")
    print(f"Валидных: {valid_count}")
    print(f"Невалидных: {len(data) - valid_count}")

    visualize_anomalies(anomaly_counter, valid_count)

if __name__ == "__main__":
    asyncio.run(main_async("dt.json"))