import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Importa as nossas classes e configurações personalizadas
from model import HeteroGNN
import config

# Importa ferramentas específicas da PyG
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

# --- Funções de Métricas Adicionais ---

def r_m_squared(y_true, y_pred):
    """
    Calcula a métrica r_m^2, usada para validação externa de modelos QSAR.
    """
    y_true = y_true.numpy().flatten()
    y_pred = y_pred.numpy().flatten()

    # Coeficiente de correlação de Pearson ao quadrado
    r_sq = pearsonr(y_true, y_pred)[0] ** 2

    # Coeficiente de correlação através da origem ao quadrado
    k = np.sum(y_true * y_pred) / np.sum(y_pred * y_pred)
    r_o_sq = 1 - (np.sum((y_true - k * y_pred)**2) / np.sum((y_true - y_true.mean())**2))

    # Cálculo final da métrica r_m^2
    if (r_sq - r_o_sq) < 0:
        rm2 = 0 # rm2 não pode ser calculado se r_o_sq > r_sq
    else:
        rm2 = r_sq * (1 - np.sqrt(r_sq - r_o_sq))
        
    return rm2

def concordance_index(y_true, y_pred):
    """
    Calcula o Índice de Concordância (CI).
    """
    y_true = y_true.numpy().flatten()
    y_pred = y_pred.numpy().flatten()
    
    # Usa um método de contagem de pares para calcular o CI
    concordant_pairs = 0
    discordant_pairs = 0
    
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
               (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                concordant_pairs += 1
            elif (y_true[i] > y_true[j] and y_pred[i] < y_pred[j]) or \
                 (y_true[i] < y_true[j] and y_pred[i] > y_pred[j]):
                discordant_pairs += 1
                
    if concordant_pairs + discordant_pairs == 0:
        return 0.5 # Retorna 0.5 se não houver pares comparáveis
        
    return concordant_pairs / (concordant_pairs + discordant_pairs)


@torch.no_grad()
def test(model, test_loader, device):
    """
    Executa a avaliação final do modelo no conjunto de teste.
    """
    model.eval()
    all_preds = []
    all_ground_truth = []

    for batch in tqdm(test_loader, desc="A avaliar no conjunto de teste"):
        batch = batch.to(device)
        
        if ('protein', 'rev_interacts_with', 'drug') not in batch.edge_types:
             batch = T.ToUndirected()(batch)

        pred = model(batch.x_dict, batch.edge_index_dict)
        ground_truth = batch['drug', 'interacts_with', 'protein'].edge_attr.squeeze()
        
        all_preds.append(pred.cpu())
        all_ground_truth.append(ground_truth.cpu())

    return torch.cat(all_preds), torch.cat(all_ground_truth)

def calculate_metrics(preds, ground_truth):
    """
    Calcula e retorna um dicionário de métricas de avaliação.
    """
    mse = torch.nn.functional.mse_loss(preds, ground_truth).item()
    rmse = np.sqrt(mse)
    p_corr, _ = pearsonr(preds.numpy(), ground_truth.numpy())
    ci = concordance_index(ground_truth, preds)
    rm2 = r_m_squared(ground_truth, preds)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'Pearson R': p_corr,
        'Concordance Index': ci,
        'r_m_squared': rm2
    }
    return metrics

def save_results(metrics, preds, ground_truth):
    """
    Guarda as métricas e o gráfico de dispersão.
    """
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "final_metrics.txt", "w") as f:
        f.write("Métricas de Avaliação Final no Conjunto de Teste\n")
        f.write("="*50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Métricas guardadas em {results_dir / 'final_metrics.txt'}")

    plt.figure(figsize=(8, 8))
    plt.scatter(ground_truth, preds, alpha=0.5, label=f'Pearson R = {metrics["Pearson R"]:.2f}')
    plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], color='red', linestyle='--', label='Linha Ideal')
    plt.title('Previsões vs. Valores Reais (Conjunto de Teste)')
    plt.xlabel('Afinidade Real')
    plt.ylabel('Afinidade Prevista')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "prediction_plot.png")
    print(f"Gráfico de dispersão guardado em {results_dir / 'prediction_plot.png'}")
    plt.close()


# --- Bloco Principal de Execução ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"A usar o dispositivo: {device}")

    # 1. Carregar o Grafo de Teste
    print("A carregar o grafo de teste...")
    test_graph = torch.load(config.PROCESSED_DATA_DIR / 'test_graph.pt')
    test_loader = DataLoader([test_graph], batch_size=1)

    # 2. Inicializar o Modelo
    model = HeteroGNN(
        hetero_metadata=test_graph.metadata(),
        drug_in_channels=test_graph['drug'].num_features,
        protein_in_channels=test_graph['protein'].num_features,
        hidden_channels=128,
        num_heads=4,
    ).to(device)

    # 3. Carregar os Pesos do Melhor Modelo
    print("A carregar os pesos do melhor modelo treinado...")
    model_path = config.MODEL_DIR / 'best_model.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 4. Executar a Avaliação
    predictions, ground_truth = test(model, test_loader, device)

    # 5. Calcular e Guardar os Resultados
    final_metrics = calculate_metrics(predictions, ground_truth)
    save_results(final_metrics, predictions, ground_truth)
    
    print("\n--- Avaliação Final Concluída ---")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")
