import pandas as pd
from tqdm import tqdm
from RAGSystem import RAGSystem

MAX_QUESTIONS = 1000
TOP_K_ANSWERS = 2

def seed():
    rag = RAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model="llama3.1",
        ollama_url="http://localhost:11434",
        qdrant_url="http://localhost:6333",
        collection_name="stackoverflow"
    )

    questions = pd.read_csv("data/Questions.csv", encoding="latin-1")
    answers = pd.read_csv("data/Answers.csv", encoding="latin-1")
    tags = pd.read_csv("data/Tags.csv", encoding="latin-1")

    answers_by_question = (
        answers
        .sort_values("Score", ascending=False)
        .groupby("ParentId")
    )

    tags_by_question = tags.groupby("Id")["Tag"].apply(list)

    documents = []
    metadata = []

    for idx, row in tqdm(questions.iterrows(), total=len(questions)):
        if idx >= MAX_QUESTIONS:
            break

        qid = int(row["Id"])

        doc_parts = [
            f"Question Title:\n{row['Title']}",
            f"\nQuestion Body:\n{row['Body']}"
        ]

        if qid in answers_by_question.groups:
            top_answers = answers_by_question.get_group(qid).head(TOP_K_ANSWERS)
            for i, ans in enumerate(top_answers.itertuples(), start=1):
                doc_parts.append(f"\nAnswer {i}:\n{ans.Body}")

        question_tags = [str(t) for t in tags_by_question.get(qid, []) if pd.notna(t)]
        if question_tags:
            doc_parts.append(f"\nTags: {', '.join(question_tags)}")

        document_text = "\n".join(doc_parts)
        documents.append(document_text)

        metadata.append({
            "question_id": qid,
            "score": int(row.get("Score", 0)),
            "tags": question_tags,
            "has_answers": qid in answers_by_question.groups
        })

    rag.add_documents(
        documents=documents,
        metadata=metadata,
        ids=[m["question_id"] for m in metadata]
    )

    print("✅ StackOverflow seeding complete")


if __name__ == "__main__":
    seed()
