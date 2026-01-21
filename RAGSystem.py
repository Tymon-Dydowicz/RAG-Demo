import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json
from datetime import datetime
import requests


class RAGSystem:
    def __init__(self, 
                 embedding_model_name='all-MiniLM-L6-v2',
                 llm_model='llama3.1',
                 ollama_url='http://localhost:11434'):
        print(f"[{self._timestamp()}] Initializing RAG System...")
        
        print(f"[{self._timestamp()}] Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        print(f"[{self._timestamp()}] Initializing FAISS vector store (dimension: {self.embedding_dim})")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.documents = []
        self.metadata = []
        
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        
        self._check_ollama()
        
        print(f"[{self._timestamp()}] RAG System initialized successfully!")
        print(f"[{self._timestamp()}] Using LOCAL LLM: {llm_model} via Ollama\n")
    
    def _timestamp(self):
        return datetime.now().strftime("%H:%M:%S")
    
    def _check_ollama(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.llm_model in model_names or any(self.llm_model in name for name in model_names):
                    print(f"[{self._timestamp()}] ✓ Ollama server is running")
                    print(f"[{self._timestamp()}] ✓ Model '{self.llm_model}' is available")
                else:
                    print(f"[{self._timestamp()}] ⚠ WARNING: Model '{self.llm_model}' not found")
                    print(f"[{self._timestamp()}] Available models: {', '.join(model_names)}")
                    print(f"[{self._timestamp()}] Run: ollama pull {self.llm_model}")
            else:
                print(f"[{self._timestamp()}] ⚠ WARNING: Ollama server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"[{self._timestamp()}] ⚠ WARNING: Cannot connect to Ollama server at {self.ollama_url}")
            print(f"[{self._timestamp()}] Make sure Ollama is installed and running:")
            print(f"[{self._timestamp()}]   1. Install: https://ollama.com")
            print(f"[{self._timestamp()}]   2. Start: ollama serve")
            print(f"[{self._timestamp()}]   3. Pull model: ollama pull {self.llm_model}")
        except Exception as e:
            print(f"[{self._timestamp()}] ⚠ WARNING: Error checking Ollama: {e}")
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        print(f"[{self._timestamp()}] Adding {len(documents)} documents to vector store...")
        
        start_idx = len(self.documents)
        self.documents.extend(documents)
        if metadata is None:
            metadata = [{"id": start_idx + i, "source": "unknown"} for i in range(len(documents))]
        self.metadata.extend(metadata)
        
        print(f"[{self._timestamp()}] Generating embeddings...")
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            batch_size=32
        )
        embeddings = np.array(embeddings).astype('float32')
        
        self.index.add(embeddings)
        
        print(f"[{self._timestamp()}] Successfully added {len(documents)} documents")
        print(f"[{self._timestamp()}] Total documents in system: {len(self.documents)}\n")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        print(f"[{self._timestamp()}] Retrieving top {top_k} documents for query: '{query}'")
        
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    "rank": i + 1,
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(dist),
                    "similarity": 1 / (1 + float(dist))
                })
        
        print(f"[{self._timestamp()}] Retrieved {len(results)} relevant documents\n")
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        print(f"[{self._timestamp()}] Generating answer using LOCAL LLM ({self.llm_model})...")
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[Document {i}]\n{doc['document']}")
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.
                    If the answer cannot be found in the context, say so.

                    Context:
                    {context_text}

                    Question: {query}

                    Provide an answer based on the context above."""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 600
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                answer = response.json()['response'].strip()
            else:
                answer = f"Error: Ollama returned status code {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            answer = "Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)"
        except requests.exceptions.Timeout:
            answer = "Error: Request to Ollama timed out. The model might be too large or slow."
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        print(f"[{self._timestamp()}] Answer generated\n")
        return answer
    
    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> Dict:
        print("=" * 80)
        print(f"[{self._timestamp()}] NEW QUERY")
        print("=" * 80)
        
        retrieved_docs = self.retrieve(question, top_k)
        
        answer = self.generate_answer(question, retrieved_docs)
        
        result = {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "timestamp": self._timestamp()
        }
        
        if verbose:
            self.print_result(result)
        
        return result
    
    def print_result(self, result: Dict):
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"\n📝 QUESTION:\n{result['question']}\n")
        print(f"💡 ANSWER:\n{result['answer']}\n")
        print(f"📚 RETRIEVED SOURCES ({len(result['retrieved_documents'])}):")
        print("-" * 80)
        
        for doc in result['retrieved_documents']:
            print(f"\n[Rank {doc['rank']}] Similarity: {doc['similarity']:.4f}")
            print(f"Source: {doc['metadata'].get('source', 'unknown')}")
            preview = doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document']
            print(f"Content: {preview}")
        
        print("\n" + "=" * 80 + "\n")
    
    def save_index(self, filepath: str = "rag_index.faiss"):
        faiss.write_index(self.index, filepath)
        
        data = {
            "documents": self.documents,
            "metadata": self.metadata
        }
        with open(filepath + ".json", "w") as f:
            json.dump(data, f)
        
        print(f"[{self._timestamp()}] Index saved to {filepath}")
    
    def load_index(self, filepath: str = "rag_index.faiss"):
        self.index = faiss.read_index(filepath)
        
        with open(filepath + ".json", "r") as f:
            data = json.load(f)
        
        self.documents = data["documents"]
        self.metadata = data["metadata"]
        
        print(f"[{self._timestamp()}] Index loaded from {filepath}")