from RAGSystem import RAGSystem

def main():
    print("\n" + "=" * 80)
    print("RAG SYSTEM DEMONSTRATION - FULLY LOCAL (No APIs!)")
    print("=" * 80 + "\n")
    
    rag = RAGSystem(
        embedding_model_name='all-MiniLM-L6-v2',
        llm_model='llama3.1',
        ollama_url='http://localhost:11434',
        qdrant_url='http://localhost:6333',
        collection_name='stackoverflow'
    )
    
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - Ask questions about Python!")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 80 + "\n")
    
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