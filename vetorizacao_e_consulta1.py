import csv
import json
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import os


class SimilaritySearch:
    def __init__(self, data: List[Dict[str, str]]):
        """Inicializa a classe, carregando dados e modelo uma única vez."""
        self.data = data
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.reclamacoes = self.extract_reclamacoes()
        self.reclamacoes_vetorizadas = self.vectorize_texts(self.reclamacoes)

    def extract_reclamacoes(self) -> List[str]:
        """Extrai as reclamações de self.data."""
        return [item['reclamacao'] for item in self.data]

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

    def run_queries(self, consultas: List[str]) -> List[Dict[str, str]]:
        """
        Executa múltiplas consultas e retorna os resultados, incluindo:
        frase consultada, reclamação mais similar, NIP, Demanda, Protocolo, e o tempo de pesquisa.
        """
        resultados = []
        for consulta in consultas:
            start_time = time.time()
            resultado = self.find_most_similar(consulta)
            end_time = time.time()
            tempo_pesquisa = end_time - start_time
            resultados.append({
                "consulta": consulta,
                "reclamacao_mais_similar": resultado['reclamacao'],
                "nip": resultado['nip'],
                "demanda": resultado['demanda'],
                "protocolo": resultado['protocolo'],
                "tempo_pesquisa": tempo_pesquisa
            })
        return resultados


def save_results_to_csv(resultados: List[Dict[str, str]], filename: str = "resultados_consultas.csv") -> None:
    """
    Salva os resultados das consultas em um arquivo CSV na mesma pasta do script.
    Se o arquivo já existir, os novos resultados são adicionados ao final.
    """
    # Caminho completo do arquivo na mesma pasta do script
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    # Verifica se o arquivo já existe
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["consulta", "reclamacao_mais_similar", "nip", "demanda", "protocolo", "tempo_pesquisa"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Escrever cabeçalho apenas se o arquivo não existir
        if not file_exists:
            writer.writeheader()

        # Escrever cada resultado no arquivo
        for resultado in resultados:
            writer.writerow(resultado)


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
    resultados = search_engine.run_queries(consultas)

    save_results_to_csv(resultados)


if __name__ == "__main__":
    main()
