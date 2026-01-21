from PrjDataset import SAMPLE_DATASET
from RAGSystem import RAGSystem

def main():
    print("\n" + "=" * 80)
    print("RAG SYSTEM DEMONSTRATION - FULLY LOCAL (No APIs!)")
    print("=" * 80 + "\n")
    
    rag = RAGSystem(
        embedding_model_name='all-MiniLM-L6-v2',
        llm_model='llama3.1',
        ollama_url='http://localhost:11434'
    )
    
    documents = [item["text"] for item in SAMPLE_DATASET]
    metadata = [{"id": i, "source": item["source"]} for i, item in enumerate(SAMPLE_DATASET)]
    
    rag.add_documents(documents, metadata)
    
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - Ask questions about Python!")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 80 + "\n")
    
    sample_questions = [
        "What is Python?",
        "How do I handle errors in Python?",
        "What are Python decorators?",
        "What is NumPy used for?",
        "What's the difference between Flask and Django?",
        "How do I work with files in Python?"
    ]
    
    print("Sample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")
    print()
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            result = rag.query(question, top_k=3, verbose=True)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()