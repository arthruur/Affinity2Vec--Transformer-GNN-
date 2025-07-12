# **Affinity2Vec (Transformer \+ GNN): Predição de Afinidade Droga-Alvo**

Este projeto implementa uma pipeline avançada de aprendizado de máquina para prever a afinidade de ligação (binding affinity) entre drogas (compostos químicos) e alvos (proteínas). A abordagem combina o poder dos **Transformers** para entender as sequências moleculares, **Graph Neural Networks (GNNs)** para capturar a topologia da rede de interações, e o **XGBoost** como um regressor final robusto.  
O objetivo é superar as limitações de modelos que analisam drogas e proteínas isoladamente, criando representações (embeddings) que são enriquecidas tanto pelo contexto da sequência quanto pela vizinhança no grafo de interações e similaridades.

## **Metodologia**

A pipeline é dividida em três estágios principais:

1. **Geração de Embeddings de Sequência com Transformers:** Utilizamos modelos Transformer pré-treinados de última geração, como ChemBERTa para as drogas (a partir de SMILES) e ProtBERT para as proteínas (a partir de sequências de aminoácidos), para gerar embeddings de alta qualidade que capturam as propriedades bioquímicas intrínsecas de cada entidade.  
2. **Enriquecimento de Embeddings com GNNs:** Os embeddings gerados pelos Transformers são usados como características iniciais para os nós de um grafo heterogêneo. Este grafo modela as complexas relações do sistema, incluindo:  
   * Similaridade Droga-Droga  
   * Similaridade Proteína-Proteína  
   * Interações conhecidas Droga-Proteína  
     Uma GNN (como a Heterogeneous Graph Attention Network \- HAN) é então treinada sobre este grafo para refinar os embeddings, propagando informações através da rede e criando representações finais que são "conscientes" do contexto global.  
3. **Predição Final com XGBoost:** Os embeddings finais e enriquecidos para cada par droga-proteína são concatenados e utilizados para treinar um modelo XGBoost. Este modelo de Gradient Boosting é altamente eficaz para capturar relações não-lineares complexas nos dados e realizar a predição final do valor de afinidade.

## **Estrutura do Projeto**

O projeto é organizado na seguinte estrutura de diretórios para garantir modularidade e reprodutibilidade:  
affinity2vec\_project/  
├── data/  
│   ├── raw/         \# Dados originais e intocados  
│   └── processed/   \# Dados limpos e prontos para uso (treino/teste)  
│  
├── models/          \# Modelos treinados e outros artefatos (scaler, etc.)  
│  
├── notebooks/       \# Notebooks para exploração e prototipagem  
│  
├── results/         \# Gráficos, métricas e predições finais  
│  
├── src/             \# Código-fonte principal da pipeline  
│   ├── config.py    \# Arquivo central de configurações e hiperparâmetros  
│   ├── data\_loader.py \# Scripts para carregar e processar dados  
│   ├── dataset.py   \# Definição de Datasets PyTorch e do grafo HeteroData  
│   ├── model.py     \# Arquitetura do modelo GNN  
│   ├── utils.py     \# Funções auxiliares e métricas de avaliação  
│   ├── train\_gnn.py \# Script para treinar o modelo GNN  
│   ├── generate\_embeddings.py \# Script para gerar embeddings com a GNN treinada  
│   └── train\_xgboost.py \# Script para treinar o regressor XGBoost final  
│  
├── .gitignore       \# Arquivos e diretórios a serem ignorados pelo Git  
├── README.md        \# Este arquivo  
└── requirements.txt \# Dependências do projeto

## **Instalação**

1. Clone o repositório:  
   git clone https://\[URL-DO-SEU-REPOSITORIO\]/affinity2vec\_project.git  
   cd affinity2vec\_project

2. Crie um ambiente virtual (recomendado):  
   python \-m venv venv  
   source venv/bin/activate  \# No Windows: venv\\Scripts\\activate

3. Instale as dependências necessárias:  
   pip install \-r requirements.txt

## **Como Executar a Pipeline**

A pipeline foi projetada para ser executada em uma sequência de passos claros e independentes.

### **Passo 0: Configuração**

Antes de executar, verifique o arquivo src/config.py. Ele contém todos os caminhos, nomes de arquivos e hiperparâmetros importantes. Ajuste-o conforme necessário para o seu ambiente.

### **Passo 1: Preparação dos Dados**

Execute o script de carregamento para processar os dados brutos e criar os conjuntos de treino e teste.  
python src/data\_loader.py

### **Passo 2: Treinamento do Modelo GNN**

Este script treina a Graph Neural Network usando os embeddings do Transformer como features iniciais e salva o modelo treinado em models/.  
python src/train\_gnn.py

### **Passo 3: Geração dos Embeddings Finais**

Com a GNN treinada, gere os embeddings finais (enriquecidos pelo grafo) para todos os dados.  
python src/generate\_embeddings.py

### **Passo 4: Treinamento do Regressor XGBoost**

Finalmente, treine o modelo XGBoost usando os embeddings gerados no passo anterior. Este script também avaliará o modelo e salvará as métricas e gráficos em results/.  
python src/train\_xgboost.py

## **Configuração**

O arquivo src/config.py é o centro de controle do projeto. Centralizar as configurações neste arquivo permite modificar facilmente parâmetros como taxas de aprendizado, tamanhos de lote, dimensões de embedding e caminhos de arquivos sem precisar alterar o código da lógica principal. Esta é uma boa prática que torna os experimentos mais rápidos e organizados.