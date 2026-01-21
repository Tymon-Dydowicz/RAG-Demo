import os
import numpy as np
import faiss
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, VectorStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json
from datetime import datetime
import requests
from tqdm import tqdm


class RAGSystem:
    def __init__(self, 
                 embedding_model_name='all-MiniLM-L6-v2',
                 llm_model='llama3.1',
                 ollama_url='http://localhost:11434',
                 qdrant_url='http://localhost:6333',
                 collection_name='stackoverflow'
                ):
        print(f"[{self._timestamp()}] Initializing RAG System...")
        
        print(f"[{self._timestamp()}] Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        print(f"[{self._timestamp()}] Connecting to Qdrant at {qdrant_url}")
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = collection_name
        self._setup_collection()
        
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
    
    def _setup_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        
        if self.collection_name not in collections:
            print(f"[{self._timestamp()}] Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"[{self._timestamp()}] Collection exists: {self.collection_name}")
        
        count = self.client.count(collection_name=self.collection_name).count
        print(f"[{self._timestamp()}] Vectors in collection: {count}")
    
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
    
    def add_documents(self, documents: List[str], metadata: List[Dict], ids: List[int] = None, batch_size=100):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            print(f"[{self._timestamp()}] Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"[{self._timestamp()}] Collection exists: {self.collection_name}")

        count = self.client.count(collection_name=self.collection_name).count
        print(f"[{self._timestamp()}] Vectors in collection before adding: {count}")

        print(f"[{self._timestamp()}] Generating embeddings...")
        embeddings = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding"):
            batch = documents[i:i+batch_size]
            batch_emb = self.embedding_model.encode(batch)
            embeddings.extend(batch_emb)

        points = []
        for i, (doc, emb, meta) in enumerate(zip(documents, embeddings, metadata)):
            point_id = ids[i] if ids is not None else i
            points.append(PointStruct(
                id=point_id,
                vector=emb.tolist(),
                payload={**meta, "text": doc}
            ))

        print(f"[{self._timestamp()}] Uploading to Qdrant...")
        for i in tqdm(range(0, len(points), batch_size), desc="Upload"):
            batch = points[i:i+batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        count = self.client.count(collection_name=self.collection_name).count
        print(f"[{self._timestamp()}] Total vectors after upload: {count}\n")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        print(f"[{self._timestamp()}] Searching for: '{query}'")
        rankCounter = 1
        
        query_emb = self.embedding_model.encode([query])[0]
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb.tolist(),
            limit=top_k,
            
        )
        
        retrieved = []
        for i, hit in enumerate(results, 1):
            _index, content = hit

            for payload in [p.payload for p in content]:
                retrieved.append({
                    "rank": rankCounter,
                    "score": payload.get('score', 0),
                    "document": payload.get('text', 'No Text Found!'),
                    "metadata": {k: v for k, v in payload.items() if k != 'text'},
                })
                rankCounter += 1
        
        print(f"[{self._timestamp()}] Found {len(retrieved)} results\n")
        return retrieved
    
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
        print(f"📚 RETRIEVED SOURCES ({len(result['retrieved_documents'])}):")
        for doc in result['retrieved_documents']:
            print(f"\n[Rank {doc['rank']}] Score: {doc['score']}")
            print(f"Tags: {doc['metadata'].get('tags', 'N/A')}")
            print(f"Has Answers: {doc['metadata'].get('has_answers', 'false')}")
            preview = doc['document'][:500] + "..." if len(doc['document']) > 500 else doc['document']
            print(f"Content: {preview}")
        print("-" * 80)
        print(f"💡 ANSWER:\n{result['answer']}\n")
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