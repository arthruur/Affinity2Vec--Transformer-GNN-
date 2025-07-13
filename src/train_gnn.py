import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Importa as nossas classes e configurações personalizadas
from model import HeteroGNN
import config

# Importa ferramentas específicas da PyG
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader # Usaremos um DataLoader mais simples

def train(model, train_loader, optimizer, criterion, device):
    """
    Executa uma época de treino completa.
    O modelo aprende com base no grafo de treino.
    """
    model.train()
    total_loss = 0
    # O nosso loader irá iterar sobre um único item: o grafo de treino completo.
    for batch in tqdm(train_loader, desc="Época de Treino"):
        optimizer.zero_grad()
        
        # Move o grafo inteiro para o dispositivo (GPU/CPU)
        batch = batch.to(device)
        
        # Garante que o grafo é bidirecional para a troca de mensagens
        # O ToUndirected() cria as arestas 'rev_interacts_with', etc.
        if ('protein', 'rev_interacts_with', 'drug') not in batch.edge_types:
             batch = T.ToUndirected()(batch)

        # O modelo faz a previsão para TODAS as arestas de interação no grafo
        pred = model(batch.x_dict, batch.edge_index_dict)
        
        # Pega nos valores de afinidade reais, que estão guardados no 'edge_attr'
        ground_truth = batch['drug', 'interacts_with', 'protein'].edge_attr.squeeze()
        
        # Calcula o erro (perda)
        loss = criterion(pred, ground_truth)
        
        # Backpropagation para aprender
        loss.backward()
        optimizer.step()
        
        # Acumula a perda
        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Executa a validação do modelo no grafo de validação.
    Não há treino aqui, apenas avaliação.
    """
    model.eval()
    total_loss = 0
    for batch in tqdm(val_loader, desc="Validação"):
        batch = batch.to(device)
        
        if ('protein', 'rev_interacts_with', 'drug') not in batch.edge_types:
             batch = T.ToUndirected()(batch)

        pred = model(batch.x_dict, batch.edge_index_dict)
        ground_truth = batch['drug', 'interacts_with', 'protein'].edge_attr.squeeze()
        loss = criterion(pred, ground_truth)
        total_loss += loss.item()

    return total_loss / len(val_loader)


# --- Bloco Principal de Execução ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"A usar o dispositivo: {device}")

    # 1. Carregar os Dados JÁ DIVIDIDOS
    # Esta é a principal mudança: carregamos os ficheiros .pt que o dataset.py criou.
    print("A carregar os grafos de treino e validação pré-divididos...")
    train_graph = torch.load(config.PROCESSED_DATA_DIR / 'train_graph.pt')
    val_graph = torch.load(config.PROCESSED_DATA_DIR / 'val_graph.pt')

    # 2. Criar DataLoaders
    # Como cada grafo já contém apenas as suas próprias interações e nós,
    # podemos usar um DataLoader simples. O batch_size=1 significa que estamos
    # a processar o grafo inteiro de uma só vez em cada época.
    train_loader = DataLoader([train_graph], batch_size=1, shuffle=True)
    val_loader = DataLoader([val_graph], batch_size=1, shuffle=False)

    # 3. Inicializar o Modelo, Otimizador e Função de Perda
    # As dimensões de entrada são retiradas diretamente do grafo de treino.
    model = HeteroGNN(
        hetero_metadata=train_graph.metadata(),
        drug_in_channels=train_graph['drug'].num_features,
        protein_in_channels=train_graph['protein'].num_features,
        hidden_channels=128, # Hiperparâmetro que pode ser ajustado
        num_heads=4,         # Hiperparâmetro que pode ser ajustado
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss() # Erro Quadrático Médio, ideal para regressão

    # 4. Loop de Treino
    best_val_loss = float('inf')
    epochs = 50 # Número de vezes que o modelo verá os dados de treino

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Época: {epoch:02d}, Perda de Treino: {train_loss:.4f}, Perda de Validação: {val_loss:.4f}")

        # 5. Guardar o Melhor Modelo
        # Se a perda de validação desta época for a mais baixa que já vimos,
        # guardamos os pesos do modelo.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Garante que o diretório para guardar o modelo existe
            Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_DIR / 'best_model.pt')
            print(f"*** Novo melhor modelo guardado com perda de validação: {val_loss:.4f} ***")

    print("\nTreino concluído!")
    print(f"Melhor perda de validação alcançada: {best_val_loss:.4f}")
    print(f"O melhor modelo está guardado em: {config.MODEL_DIR / 'best_model.pt'}")
