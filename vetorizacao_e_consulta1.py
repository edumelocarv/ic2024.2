import json
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time


class SimilaritySearch:
    def __init__(self, data: List[Dict[str, str]]):
        """Inicializa a classe, carregando dados e modelo uma única vez."""
        self.data = data
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.reclamacoes = self.extract_reclamacoes(self.data)
        self.reclamacoes_vetorizadas = self.vectorize_texts(self.reclamacoes)

    @staticmethod
    def extract_reclamacoes(data: List[Dict[str, str]]) -> List[str]:
        """Extrai as reclamações de uma lista de dicionários."""
        return [item['reclamacao'] for item in data]

    def vectorize_texts(self, texts: List[str]) -> np.ndarray:
        """Vetoriza uma lista de textos usando o modelo SBERT e retorna os vetores resultantes."""
        return self.model.encode(texts)

    def calculate_similarity(self, vetor_consulta: np.ndarray) -> np.ndarray:
        """Calcula a similaridade de cosseno entre a consulta e as reclamações vetorizadas."""
        return cosine_similarity([vetor_consulta], self.reclamacoes_vetorizadas).flatten()

    def find_most_similar(self, consulta: str) -> Dict[str, str]:
        """
        Recebe uma consulta de texto, calcula a similaridade com as reclamações
        e retorna a reclamação mais similar e suas informações.
        """
        vetor_consulta = self.vectorize_texts([consulta])[0]
        similarities = self.calculate_similarity(vetor_consulta)
        index = np.argmax(similarities)
        return self.data[index]

    def run_queries(self, consultas: List[str]) -> None:
        """Executa múltiplas consultas e exibe as reclamações mais similares para cada uma."""
        for consulta in consultas:
            start_time = time.time()
            resultado = self.find_most_similar(consulta)
            end_time = time.time()
            print(f"\nConsulta: {consulta}")
            print(f"Reclamação mais similar: {resultado['reclamacao']}")
            print(f"NIP: {resultado['nip']}")
            print(f"Demanda: {resultado['demanda']}")
            print(f"Protocolo: {resultado['protocolo']}")
            print(f"Tempo de pesquisa: {end_time - start_time:.4f} segundos")


def load_data(file_path: str) -> List[Dict[str, str]]:
    """
    Carrega dados de um arquivo JSON ou CSV e retorna uma lista de dicionários com informações das reclamações.
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding="utf-8")
        data = df.to_dict(orient="records")
    else:
        raise ValueError("Formato de arquivo não suportado. Use JSON ou CSV.")
    
    return data

def main() -> None:
    file_path = 'dados.json'
    data = load_data(file_path)
    consultas = ["são luis", "solicitação de exame negado", "atendimento psicológico"]
    search_engine = SimilaritySearch(data)
    search_engine.run_queries(consultas)


if __name__ == "__main__":
    main()
