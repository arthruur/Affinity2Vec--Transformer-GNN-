# **Affinity2Vec (Transformer \+ GNN): Predição de Afinidade Droga-Alvo End-to-End**

Este projeto implementa uma pipeline de aprendizado de máquina para prever a afinidade de ligação (binding affinity) entre drogas e alvos proteicos. A abordagem utiliza uma **Graph Neural Network (GNN) End-to-End**, que aprende a partir de características moleculares geradas por **Transformers** e da topologia de um grafo de interações complexas.  
O objetivo é criar um modelo único e otimizado que não apenas aprende representações (embeddings) enriquecidas pelo contexto do grafo, mas também realiza a predição final da afinidade, superando as limitações de modelos que analisam as entidades de forma isolada ou em múltiplas etapas não otimizadas.

## **Metodologia**

A pipeline é dividida em dois estágios principais:

1. **Geração de Embeddings Iniciais com Transformers:** Utilizamos modelos Transformer pré-treinados, como ChemBERTa (para SMILES de drogas) e ProtBERT/ESM (para sequências de aminoácidos de proteínas), para gerar embeddings de alta qualidade. Estes embeddings servem como as características iniciais dos nós no nosso grafo, capturando as propriedades bioquímicas intrínsecas de cada entidade.  
2. **Treinamento e Predição com GNN End-to-End:** Os embeddings iniciais são usados como features de nós em um grafo heterogêneo, que é então treinado diretamente para a tarefa de regressão.

### **Estrutura do Grafo Heterogêneo**

O coração do modelo é um grafo heterogêneo que captura diferentes tipos de entidades e relações:

* **Nós (Nodes):**  
  * drug: Representa cada composto químico único no conjunto de dados.  
  * protein: Representa cada alvo proteico único.  
* **Arestas (Edges):** As relações entre os nós são modeladas por diferentes tipos de arestas, permitindo que a GNN aprenda com múltiplos contextos:  
  * ('drug', 'interacts\_with', 'protein'): A aresta principal do problema. Conecta uma droga a uma proteína com base em um valor de afinidade conhecido. É esta relação que o modelo aprende a prever.  
  * ('drug', 'similar\_to', 'drug'): Conecta duas drogas que são estruturalmente similares ( alta similaridade de Tanimoto baseada nos seus SMILES).  
  * ('protein', 'similar\_to', 'protein'): Conecta duas proteínas que são estruturalmente similares (alta similaridade de alinhamento de sequência como Smith-Waterman).  
  * ('drug', 'semantic\_similar\_to', 'drug'): Conecta duas drogas que são semanticamente próximas no espaço de embedding inicial (alta similaridade de cosseno).  
  * ('protein', 'semantic\_similar\_to', 'protein'): Conecta duas proteínas que são semanticamente próximas no espaço de embedding inicial.

### **Fluxo de Aprendizagem na GNN**

* **Message Passing:** As camadas da GNN (ex: GATConv) propagam e agregam informações através de todos os tipos de arestas, refinando os embeddings dos nós para que eles se tornem "conscientes" do seu contexto global no grafo.  
* **Predição Direta:** Uma "cabeça" de regressão (um MLP decoder) é integrada ao final da arquitetura da GNN. Ela pega os embeddings finais e enriquecidos de um par droga-proteína e prevê diretamente o valor de afinidade.  
* **Otimização de Ponta a Ponta:** O erro da predição final é retropropagado através de toda a rede, ajustando tanto os pesos do decoder quanto os das camadas da GNN. Isto garante que os embeddings sejam otimizados especificamente para a tarefa de prever a afinidade.

## **Estrutura do Projeto**

O projeto é organizado na seguinte estrutura de diretórios para garantir modularidade e reprodutibilidade:  
affinity2vec\_project/  
├── data/  
│   ├── raw/         \# Dados originais e intocados  
│   └── processed/   \# Grafos processados e prontos para uso (train/val/test)  
│  
├── models/          \# Modelos GNN treinados (.pt)  
│  
├── results/         \# Gráficos, métricas e predições finais  
│  
├── src/             \# Código-fonte principal da pipeline  
│   ├── config.py    \# Arquivo central de configurações e hiperparâmetros  
│   ├── dataset.py   \# Script para construir e salvar os grafos (train/val/test)  
│   ├── model.py     \# Arquitetura do modelo HeteroGNN  
│   ├── train.py     \# Script para treinar o modelo GNN  
│   └── evaluate.py  \# Script para avaliar o modelo GNN treinado  
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

A pipeline foi projetada para ser executada em uma sequência de três passos claros.

### **Passo 0: Configuração**

Antes de executar, verifique o arquivo src/config.py. Ele contém todos os caminhos, nomes de arquivos e hiperparâmetros importantes. Ajuste-o conforme necessário para o seu ambiente.

### **Passo 1: Construção dos Grafos**

Execute este script **uma vez** para processar os dados de interação e criar os ficheiros de grafo para treino, validação e teste.  
python src/dataset.py

### **Passo 2: Treinamento do Modelo GNN**

Este script carrega os grafos de treino e validação, treina o modelo HeteroGNN e salva a melhor versão (com base no desempenho de validação) na pasta models/.  
python src/train.py

### **Passo 3: Avaliação Final do Modelo**

Finalmente, execute este script para carregar o melhor modelo salvo e avaliá-lo no conjunto de teste. Ele salvará as métricas de desempenho e um gráfico de predição na pasta results/.  
python src/evaluate.py

## **Configuração**

O arquivo src/config.py é o centro de controle do projeto. Centralizar as configurações neste arquivo permite modificar facilmente parâmetros como taxas de aprendizado, tamanhos de lote, dimensões de embedding e caminhos de arquivos sem precisar alterar o código da lógica principal. Esta é uma boa prática que torna os experimentos mais rápidos e organizados.