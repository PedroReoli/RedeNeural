# 🧠 Classificação de Pontos com Rede Neural Simples

Este projeto demonstra o uso de uma **rede neural de camada única** para resolver um problema de classificação simples: **identificar se um ponto bidimensional está dentro ou fora de um círculo**. A rede é treinada usando duas funções de ativação diferentes: **Sigmoid** e **ReLU**, permitindo a comparação de desempenho entre elas.

---

## 📌 Visão Geral

O código realiza as seguintes etapas:

1. **Geração de dados**: cria 500 pontos aleatórios no espaço 2D.
2. **Classificação dos pontos**: define como "1" os pontos dentro de um círculo de raio 0.5.
3. **Visualização**: exibe os pontos coloridos por classe.
4. **Treinamento da rede**: comparamos o uso das funções de ativação **Sigmoid** e **ReLU**.
5. **Plotagem do erro**: exibe a evolução do erro durante o treinamento.
6. **Testes**: aplica a rede a algumas entradas e compara a saída obtida com a esperada.

---

## 📈 Visualização do Dataset

O dataset consiste em pontos gerados aleatoriamente no intervalo [-1, 1] para as duas coordenadas. A classe de cada ponto é definida com base na sua distância à origem (0,0), formando um círculo:

- **Classe 1**: ponto dentro do círculo (distância < 0.5)  
- **Classe 0**: ponto fora do círculo

O gráfico inicial mostra a distribuição dos pontos e suas respectivas classes.

---

## ⚙️ Funções de Ativação

Duas funções clássicas de ativação são implementadas:

### Sigmoid
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### Derivada da Sigmoid
```python
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

### ReLU
```python
def relu(x):
    return np.maximum(0, x)
```

### Derivada da ReLU
```python
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
```

As derivadas são essenciais para o processo de **backpropagation**, pois determinam como os pesos serão ajustados.

---

## 🧪 Treinamento da Rede Neural

A rede possui:

- **2 entradas** (as coordenadas X e Y do ponto)
- **1 neurônio na saída**
- **Sem camadas ocultas**
- **Taxa de aprendizado**: `0.5`
- **Épocas**: `200.000`

Durante o treinamento:

1. Realiza-se o **forward pass** com a função de ativação escolhida.
2. Calcula-se o **erro quadrático médio** entre a saída esperada e a obtida.
3. Aplica-se o **backpropagation**, ajustando pesos e viés com base na derivada da função de ativação.

A cada 5.000 épocas, o erro atual é exibido no console.

---

## 📊 Gráfico de Erro

Após o treinamento com as duas funções de ativação, o código plota um gráfico com a evolução do erro em cada época:

- 🔵 **Sigmoid**: linha azul sólida  
- 🔴 **ReLU**: linha vermelha tracejada

Este gráfico facilita a comparação da taxa de convergência entre as duas abordagens.

---

## 🧪 Testes Finais

Ao final do treinamento, o modelo é testado com os **10 primeiros pontos do dataset**. Para cada entrada, são exibidos:

- As coordenadas do ponto (entrada)
- A classe esperada (rótulo real)
- A saída gerada pela rede

O teste é feito separadamente para os modelos treinados com Sigmoid e ReLU.

---

## 📁 Requisitos

Para executar este projeto, você precisará do Python 3 e das seguintes bibliotecas:

```bash
pip install numpy matplotlib
```

---

## 📝 Execução

Salve o código-fonte em um arquivo chamado `neural_circle_classification.py` e execute com:

```bash
python neural_circle_classification.py
```

---

## 🎓 Objetivo Educacional

Este projeto foi desenvolvido com fins educacionais, com o objetivo de:

- Demonstrar o funcionamento básico de uma rede neural simples
- Comparar funções de ativação no aprendizado de máquina
- Promover entendimento de conceitos como **forward pass**, **backpropagation** e **gradiente descendente**

---

## 👥 Autores

Este projeto foi desenvolvido por:

- **Pedro Lucas Reis de Oliveira Sousa**  
- **Pedro Amando Gandos Citelli**
