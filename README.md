Ninfa - IA Generativa com Memória Persistente

![alt text](https://img.shields.io/badge/License-MIT-blue.svg)


![alt text](https://img.shields.io/badge/.NET-8.0-blueviolet)


![alt text](https://img.shields.io/badge/Status-Em%20Desenvolvimento-green)

Ninfa é um projeto de Inteligência Artificial Generativa que vai além da simples predição de texto. É uma implementação de um agente de IA com um "cérebro" (uma rede neural LSTM customizada) e uma "memória de longo prazo" (uma base de conhecimento persistente em disco). O sistema é projetado para aprender, refletir, adquirir novo conhecimento em tempo real e evoluir a partir de suas interações.
Sumário

    Visão Geral

    Principais Funcionalidades

    Arquitetura do Sistema

        Diagrama de Fluxo

        Análise dos Componentes

            1. O Núcleo Generativo (O Cérebro)

            2. A Memória Virtual Persistente (A Base de Conhecimento)

            3. A Camada de Serviços (As Funções Cognitivas)

            4. O Host da Aplicação (O Corpo)

    Como Começar

        Pré-requisitos

        Instalação e Configuração

    Como Usar

        Modo 1: Executando a API Web

        Modo 2: Executando o Treinamento

    Endpoints da API

    Roteiro de Desenvolvimento (Roadmap)

Visão Geral

Este projeto implementa uma rede neural LSTM (Long Short-Term Memory) capaz de gerar texto. Seu principal diferencial é a integração com um sistema de memória customizado que permite à IA:

    Armazenar conhecimento de forma persistente em um arquivo de memória virtual.

    Refletir sobre a necessidade de novas informações com base nos prompts do usuário.

    Adquirir conhecimento de fontes externas (fontes de dados na internet).

    Internalizar o novo conhecimento através de um micro-treinamento (fine-tuning).

    Adaptar-se dinamicamente, expandindo seu próprio vocabulário e a arquitetura da rede neural em tempo de execução.

Principais Funcionalidades

    🧠 Motor LSTM Customizado: Rede neural implementada do zero em C#, incluindo backpropagation.

    💾 Memória Virtual Persistente: Uma árvore binária de busca baseada em arquivo, que atua como uma base de conhecimento não-volátil.

    🌐 Aquisição de Conhecimento em Tempo Real: Conecta-se a APIs externas para buscar informações atualizadas quando seu conhecimento interno é insuficiente.

    📈 Aprendizado Contínuo: Capacidade de "internalizar" novas informações, ajustando os pesos da rede com uma baixa taxa de aprendizado.

    🧩 Vocabulário Dinâmico: Expande seu vocabulário e adapta as dimensões da rede neural em tempo real para acomodar novos conceitos.

    🚀 Execução em Modo Duplo: Pode ser executado como uma API Web (para interação) ou como uma aplicação de console (para treinamento em lote).

    💻 Aceleração por GPU (Opcional): Suporte a OpenCL para acelerar operações de matriz, com fallback para CPU.

Arquitetura do Sistema

O sistema é modular, dividido em quatro pilares principais que trabalham de forma independente e em conjunto.
Diagrama de Fluxo

Um fluxo de requisição para o endpoint /generate ilustra a sinergia entre os componentes:
code Code
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END

    
Usuário -> [API Host] -> GenerativeAIController
   |
   +--> ContextManager (Reflexão)
           |
           +--> [Memória Virtual] (Consulta conhecimento existente)
           |
           +--> KnowledgeAcquisitionService (Se necessário)
                   |
                   +--> ChatGPTService / GoogleSearchService (Busca externa)
           |
           +--> (Processa e Internaliza novo conhecimento)
           |
           +--> [Memória Virtual] (Armazena novo conhecimento)
   |
   +--> NeuralNetwork (Geração de texto com contexto enriquecido)
   |
   V
Resposta -> Usuário

  

Análise dos Componentes
1. O Núcleo Generativo (O Cérebro)

    Componente Principal: Core/NeuralNetwork.cs

    Funcionamento Independente:
    Esta classe é uma implementação autocontida de uma rede neural LSTM. Ela gerencia todas as matrizes de pesos e vieses, e contém a lógica matemática para o forward pass (prever o próximo token) e o backpropagation through time (aprender com os erros). É agnóstica em relação a como é usada; sua única função é processar tensores de entrada e produzir tensores de saída.

    Interação no Ecossistema:
    É o "músculo" computacional do sistema. O GenerativeAIController o utiliza para a geração final de texto, fornecendo-lhe um prompt (a semente). O Trainer o utiliza de forma intensiva para ajustar seus pesos com base em um grande dataset, constituindo o processo de aprendizado primário.

2. A Memória Virtual Persistente (A Base de Conhecimento)

    Componentes Principais: BinaryTreeSwapFile/BinaryTreeFileStorage.cs, BinaryTreeSwapFile/TreeNode.cs

    Funcionamento Independente:
    Este módulo implementa uma estrutura de dados de árvore binária de busca que opera diretamente sobre um único arquivo em disco. Em vez de ponteiros de memória, ele usa offsets de arquivo (long) para conectar os nós. Cada nó tem um tamanho fixo, contendo um payload de dados, os offsets dos filhos e um timestamp de último acesso. Ele fornece operações de I/O de baixo nível (leitura/escrita de nós) e um mecanismo de limpeza para expirar dados antigos.

    Interação no Ecossistema:
    Funciona como a memória de longo prazo da IA. O ContextManager é seu principal cliente, usando-o para armazenar e recuperar objetos ContextInfo (resumos de conhecimento) serializados. O Trainer também o utiliza para "semear" conhecimento inicial durante o treinamento em lote. É a espinha dorsal da persistência de conhecimento do sistema.

3. A Camada de Serviços (As Funções Cognitivas)

    Componentes Principais: Services/ (e.g., ContextManager.cs, KnowledgeAcquisitionService.cs, etc.)

    Funcionamento Independente:
    Cada serviço nesta camada tem uma responsabilidade única e bem definida:

        ChatGPTService & GoogleSearchService: Clientes de API autocontidos que atuam como os "sentidos" da IA para o mundo externo.

        KnowledgeAcquisitionService: Um orquestrador que decide qual "sentido" usar, implementando uma estratégia de busca hierárquica (tenta MCFP, se falhar, tenta Google).

        TextProcessorService: Um conjunto de ferramentas stateless para manipulação de texto (resumo, hashing, etc.).

        ContextManager: O "gerenciador da memória". Ele encapsula a lógica de quando consultar a memória, quando o conhecimento está obsoleto e quando novo conhecimento deve ser armazenado.

    Interação no Ecossistema:
    Esta camada é a cola que conecta o "cérebro" à "memória". O GenerativeAIController delega quase toda a lógica de "pensamento" para o ContextManager. Este, por sua vez, coordena os outros serviços para executar o ciclo completo de reflexão, aquisição e armazenamento de conhecimento, antes de finalmente entregar o contexto enriquecido de volta para o controlador.

4. O Host da Aplicação (O Corpo)

    Componentes Principais: Program.cs, Startup.cs, Controllers/GenerativeAIController.cs

    Funcionamento Independente:
    É uma aplicação ASP.NET Core padrão com uma característica notável: seu ponto de entrada (Program.cs) implementa uma lógica de modo duplo. Com base em um argumento de linha de comando (--train), ele decide se deve iniciar o servidor web completo ou apenas um host mínimo para executar o Trainer.

    Interação no Ecossistema:
    É o invólucro que une tudo. No modo API, ele expõe todas as funcionalidades através de endpoints HTTP e usa injeção de dependência (Startup.cs) para construir e conectar todos os outros componentes. O GenerativeAIController atua como o ponto de entrada para todas as interações do usuário, orquestrando o fluxo de dados entre as diferentes camadas.

Como Começar
Pré-requisitos

    .NET 8 SDK ou superior.

    (Opcional) Drivers de GPU com suporte a OpenCL para aceleração de hardware.

Instalação e Configuração

    Clone o repositório:
    code Bash

IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END

    
git clone https://[URL_DO_SEU_REPOSITORIO].git
cd generative

  

Restaure as dependências:
code Bash
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END

    
dotnet restore

  

Configure as chaves de API e caminhos:
Crie ou edite o arquivo appsettings.Development.json na raiz do projeto. Ele deve conter as suas chaves de API e os caminhos para os arquivos do modelo e do dataset.
code JSON

    IGNORE_WHEN_COPYING_START
    IGNORE_WHEN_COPYING_END

        
    {
      "ModelSettings": {
        "ModelDirectory": "Caminho/Para/Seu/Diretorio/De/Modelos",
        "ContextWindowSize": 10,
        "HiddenSize": 256
      },
      "TrainingSettings": {
        "DatasetPath": "Caminho/Para/Seu/dataset.txt",
        "Epochs": 10
      },
      "ApiKeys": {
        "OpenAI": "sua_chave_sk-...",
        "GoogleSearch": "sua_chave_do_google_search_api",
        "GoogleSearchCx": "seu_id_de_mecanismo_de_busca_personalizado"
      }
    }

      

Como Usar
Modo 1: Executando a API Web

Este modo inicia o servidor e expõe os endpoints para interação.
code Bash
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END

    
dotnet run

  

Após a inicialização, acesse a interface do Swagger no seu navegador (geralmente em https://localhost:7096/swagger) para testar os endpoints.
Modo 2: Executando o Treinamento

Este modo executa o Trainer como uma aplicação de console para treinar o modelo a partir de um dataset.
code Bash
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END

    
dotnet run -- --train [opções]

  

Opções de Treinamento:

    --epoch <num>: Define o número total de épocas de treinamento. (Ex: --epoch 50)

    --start-epoch <num>: Define a época inicial (útil para retomar o treinamento). (Ex: --start-epoch 11)

    --window-size <num>: Define o tamanho da janela de contexto. (Ex: --window-size 15)

    --chunk-size <num>: Define o número de linhas do dataset a serem lidas na memória por vez. (Ex: --chunk-size 500)

Endpoints da API
Endpoint	Método	Descrição
/api/generate	POST	Gera texto a partir de uma semente, usando o ciclo de reflexão.
/api/train	POST	Inicia um treinamento do modelo com o texto fornecido no corpo.
/api/summarize	POST	Resume um texto longo e armazena o resumo na memória virtual.
/api/test	POST	Avalia a perda (loss) do modelo em um conjunto de dados de teste.
/api/evaluate	POST	Gera uma continuação para um texto de entrada para avaliação qualitativa.
Roteiro de Desenvolvimento (Roadmap)

Implementar uma busca mais eficiente na BinaryTreeFileStorage (e.g., indexação).

    Status: Concluído.

    Detalhes: A busca na memória virtual foi otimizada. Em vez de uma travessia completa (O(n)), a estrutura agora se beneficia da organização inerente de uma Árvore Binária de Busca (BST). A chave de busca (o tópico do contexto) é usada para navegar pela árvore, reduzindo o tempo de busca médio para O(log n) em uma árvore balanceada, melhorando significativamente a performance da fase de "Reflexão".

Melhorar a implementação OpenCL com kernels de backpropagation.

    Status: Concluído.

    Detalhes: A aceleração por GPU foi estendida para abranger o processo de treinamento. Além dos kernels do forward pass, foram implementados kernels OpenCL para as operações do backward pass (backpropagation). Isso inclui multiplicação de matriz transposta, cálculo de derivadas de funções de ativação e acumulação de gradientes, movendo a parte mais computacionalmente intensiva do treinamento para a GPU. O resultado é uma redução drástica no tempo necessário para cada época de treinamento.

Adicionar um endpoint para visualizar o estado da memória virtual.

    Status: Concluído.

    Detalhes: Foi adicionado um novo endpoint, GET /api/memory/view, para fins de diagnóstico e observabilidade. Este endpoint realiza uma travessia em ordem (in-order traversal) na BinaryTreeFileStorage e retorna uma lista de todos os contextos atualmente armazenados, incluindo seus tópicos, timestamps e offsets no arquivo. Isso permite aos desenvolvedores inspecionar o que a IA aprendeu e memorizou ao longo do tempo.
