import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

class HeteroGNN(torch.nn.Module):
    """
    Define a arquitetura da Rede Neural de Grafo Heterogênea (HeteroGNN).

    Esta classe implementa uma GNN capaz de processar o grafo heterogêneo
    construído na fase de dataset. A arquitetura consiste em três partes principais:
    1. Camadas lineares iniciais para projetar os diferentes tipos de nós para um 
       espaço de características comum (hidden_channels).
    2. Duas camadas de convolução de grafo heterogêneas (HeteroConv) que realizam
       a troca de mensagens, usando GATConv para aprender a importância dos vizinhos.
    3. Uma "cabeça" de previsão (MLP) que pega os embeddings finais de um par 
       droga-proteína e prevê a sua afinidade.
    """
    def __init__(self, hetero_metadata, drug_in_channels, protein_in_channels, hidden_channels, num_heads=4, out_channels=1):
        """
        Inicializa o modelo HeteroGNN.

        Args:
            hetero_metadata (tuple): Metadados do grafo, contendo os tipos de nós e arestas.
                                     Necessário para a inicialização da camada HeteroConv.
            drug_in_channels (int): Dimensão das características de entrada dos nós de droga.
            protein_in_channels (int): Dimensão das características de entrada dos nós de proteína.
            hidden_channels (int): O número de canais (dimensão) no espaço latente.
            num_heads (int): O número de "cabeças de atenção" a serem usadas nas camadas GATConv.
            out_channels (int): A dimensão da saída final (1 para prever a afinidade).
        """
        super().__init__()

        # --- 1. Camadas de Projeção Inicial ---
        # Como os nós de droga e proteína têm dimensões de entrada diferentes,
        # primeiro os projetamos para um espaço de características comum (hidden_channels).
        self.drug_lin = Linear(drug_in_channels, hidden_channels)
        self.protein_lin = Linear(protein_in_channels, hidden_channels)

        # --- 2. Camadas de Convolução Heterogêneas (Message Passing) ---
        # Primeira camada HeteroConv.
        self.conv1 = HeteroConv({
            # Para cada tipo de aresta, definimos uma camada de convolução de grafo.
            # Usamos GATConv para permitir que o modelo aprenda a importância de cada vizinho.
            ('drug', 'interacts_with', 'protein'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('protein', 'rev_interacts_with', 'drug'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('drug', 'similar_to', 'drug'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('protein', 'similar_to', 'protein'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('drug', 'semantic_similar_to', 'drug'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('protein', 'semantic_similar_to', 'protein'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
        }, aggr='sum') # Agrega as mensagens de diferentes tipos de aresta por soma.

        # Segunda camada HeteroConv.
        # A entrada para esta camada é a saída da anterior (hidden_channels * num_heads).
        self.conv2 = HeteroConv({
            ('drug', 'interacts_with', 'protein'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('protein', 'rev_interacts_with', 'drug'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('drug', 'similar_to', 'drug'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('protein', 'similar_to', 'protein'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('drug', 'semantic_similar_to', 'drug'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('protein', 'semantic_similar_to', 'protein'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
        }, aggr='sum')
        
        # --- 3. Cabeça de Previsão (MLP Decoder) ---
        # Esta rede neural pega os embeddings finais concatenados de um par droga-proteína
        # e os mapeia para um único valor de afinidade.
        # A entrada é 2 * hidden_channels * num_heads porque concatenamos o embedding da droga e da proteína.
        self.decoder = torch.nn.Sequential(
            Linear((hidden_channels * num_heads) * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            Linear(128, out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        """
        Define o fluxo de dados através do modelo (forward pass).

        Args:
            x_dict (dict): Dicionário contendo as matrizes de características para cada tipo de nó.
                           Ex: {'drug': tensor, 'protein': tensor}
            edge_index_dict (dict): Dicionário contendo os índices de arestas para cada tipo de relação.

        Returns:
            torch.Tensor: Um tensor contendo o valor de afinidade previsto para cada par
                          droga-proteína na aresta 'interacts_with'.
        """
        # --- 1. Projeção Inicial ---
        # Aplica as camadas lineares para mapear as características de entrada para o espaço latente.
        x_dict = {
          'drug': self.drug_lin(x_dict['drug']),
          'protein': self.protein_lin(x_dict['protein']),
        }

        # --- 2. Message Passing ---
        # Passa os dados pela primeira camada HeteroConv e aplica uma função de ativação.
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        # Passa os dados pela segunda camada HeteroConv e aplica uma função de ativação.
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        # --- 3. Previsão (Decoding) ---
        # Para prever a afinidade, focamos nas arestas de interação.
        interaction_edge_index = edge_index_dict[('drug', 'interacts_with', 'protein')]
        
        # Obtém os embeddings finais para os nós de droga (source) e proteína (target)
        # que participam das arestas de interação.
        drug_emb = x_dict['drug'][interaction_edge_index[0]]
        protein_emb = x_dict['protein'][interaction_edge_index[1]]
        
        # Concatena os embeddings do par.
        concatenated_emb = torch.cat([drug_emb, protein_emb], dim=-1)
        
        # Usa o decoder para prever a afinidade.
        prediction = self.decoder(concatenated_emb)
        
        # Remove a última dimensão (de tamanho 1) para ter um tensor de predições.
        return prediction.squeeze(-1)

