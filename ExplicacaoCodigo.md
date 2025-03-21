# Por trás do código: Classificação de Pontos com Rede Neural

Este documento explica detalhadamente o funcionamento do código que implementa uma rede neural simples para classificar pontos como estando dentro ou fora de um círculo.

## 📚 Sumário

1. [Visão Geral](#visão-geral)
2. [Funções de Ativação](#funções-de-ativação)
3. [Geração do Dataset](#geração-do-dataset)
4. [Treinamento da Rede Neural](#treinamento-da-rede-neural)
5. [Backpropagation](#backpropagation)
6. [Comparação: Sigmoid vs ReLU](#comparação-sigmoid-vs-relu)
7. [Teste do Modelo](#teste-do-modelo)

## 🔍 Visão Geral

O código implementa uma rede neural de camada única (perceptron) para resolver um problema de classificação binária: determinar se um ponto em um espaço bidimensional está dentro ou fora de um círculo com raio 0.5 centrado na origem.

A implementação compara duas funções de ativação populares:
- **Sigmoid**: Uma função suave que mapeia valores para o intervalo (0,1)
- **ReLU (Rectified Linear Unit)**: Uma função que retorna 0 para entradas negativas e a própria entrada para valores positivos

## ⚙️ Funções de Ativação

### Sigmoid e sua Derivada

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

A função sigmoid transforma qualquer valor de entrada em um número entre 0 e 1. Ela é definida matematicamente como σ(x) = 1/(1+e^(-x)).

**Características da Sigmoid:**
- Suave e diferenciável em todos os pontos
- Saída limitada entre 0 e 1 (útil para probabilidades)
- Saturação: para valores muito grandes (positivos ou negativos), a derivada se aproxima de zero
- Sua derivada é calculada como σ(x) * (1 - σ(x))

### ReLU e sua Derivada

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
```

A função ReLU (Rectified Linear Unit) é definida como f(x) = max(0, x).

**Características da ReLU:**
- Computacionalmente eficiente
- Não satura para valores positivos (evita o problema do desaparecimento do gradiente)
- Derivada simples: 1 para x > 0 e 0 para x ≤ 0
- Problema de "neurônios mortos": neurônios podem parar de aprender se sempre receberem entradas negativas

## 📊 Geração do Dataset

```python
np.random.seed(0)  # Garante reprodutibilidade
X = np.random.uniform(-1, 1, (500, 2))  # 500 pontos aleatórios entre -1 e 1
y = (np.sqrt(X[:, 0]**2 + X[:, 1]**2) < 0.5).astype(int).reshape(-1, 1)
```

O código gera 500 pontos aleatórios em um quadrado de -1 a 1 em ambas as dimensões. Cada ponto é classificado como:
- **Classe 1 (verdadeiro)**: se estiver dentro do círculo de raio 0.5 centrado na origem
- **Classe 0 (falso)**: se estiver fora do círculo

A fórmula `np.sqrt(X[:, 0]**2 + X[:, 1]**2) < 0.5` calcula a distância euclidiana de cada ponto até a origem (0,0) e verifica se é menor que 0.5 (o raio do círculo).

A visualização do dataset é feita com matplotlib:

```python
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Spectral)
plt.title("Dataset: Pontos Dentro e Fora de um Círculo")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Classe")
plt.show()
```

## 🧠 Treinamento da Rede Neural

O treinamento é implementado na função `train_neural_network`:

```python
def train_neural_network(activation_function, activation_derivative):
    weights = np.random.randn(2, 1)  # Inicializa pesos aleatórios
    bias = np.random.randn()  # Inicializa o viés aleatoriamente
    errors = []

    for epoch in range(epochs):
        # Forward pass
        weighted_sum = np.dot(X, weights) + bias
        output = activation_function(weighted_sum)

        # Cálculo do erro
        error = y - output
        errors.append(np.mean(np.square(error)))

        # Backpropagation
        d_error = -2 * error / len(X)
        d_output = activation_derivative(weighted_sum)
        gradient = d_error * d_output

        # Atualização dos pesos e viés
        weights -= learning_rate * np.dot(X.T, gradient)
        bias -= learning_rate * np.sum(gradient)

        if epoch % 5000 == 0:
            print(f"Época {epoch}, Erro: {errors[-1]}")

    return weights, bias, errors
```

### Inicialização
- Os pesos (`weights`) são inicializados aleatoriamente com uma distribuição normal
- O viés (`bias`) também é inicializado aleatoriamente
- A lista `errors` armazenará o erro em cada época para análise posterior

### Forward Pass
1. **Soma Ponderada**: `weighted_sum = np.dot(X, weights) + bias`
   - Multiplica cada entrada pelos pesos correspondentes e soma o resultado com o viés
   - Matematicamente: z = w₁x₁ + w₂x₂ + b

2. **Ativação**: `output = activation_function(weighted_sum)`
   - Aplica a função de ativação (sigmoid ou ReLU) à soma ponderada
   - Transforma o valor linear em uma previsão não-linear

## 🔄 Backpropagation

O backpropagation é o processo de ajustar os pesos da rede com base no erro:

1. **Cálculo do Erro**: `error = y - output`
   - Diferença entre o valor esperado (y) e a saída prevista (output)

2. **Erro Quadrático Médio**: `np.mean(np.square(error))`
   - Mede a performance do modelo (menor é melhor)
   - Elevamos ao quadrado para penalizar erros maiores e evitar que erros positivos e negativos se cancelem

3. **Derivada do Erro**: `d_error = -2 * error / len(X)`
   - Derivada do erro quadrático médio em relação à saída
   - O fator -2 vem da derivada da função de erro quadrático

4. **Derivada da Ativação**: `d_output = activation_derivative(weighted_sum)`
   - Calcula a derivada da função de ativação no ponto da soma ponderada
   - Indica a taxa de mudança da saída em relação à entrada

5. **Gradiente**: `gradient = d_error * d_output`
   - Combina as derivadas pela regra da cadeia
   - Indica a direção e magnitude para ajustar os pesos

6. **Atualização dos Pesos**: 
   - `weights -= learning_rate * np.dot(X.T, gradient)`
   - `bias -= learning_rate * np.sum(gradient)`
   - Ajusta os pesos e o viés na direção oposta ao gradiente
   - O learning_rate controla o tamanho do passo de atualização

## 📈 Comparação: Sigmoid vs ReLU

O código treina dois modelos idênticos, exceto pela função de ativação:

```python
weights_sigmoid, bias_sigmoid, erros_sigmoid = train_neural_network(sigmoid, sigmoid_derivative)
weights_relu, bias_relu, erros_relu = train_neural_network(relu, relu_derivative)
```

A comparação visual é feita plotando o erro ao longo das épocas:

```python
plt.plot(range(epochs), erros_sigmoid, label="Sigmoid", color='blue')
plt.plot(range(epochs), erros_relu, label="ReLU", linestyle='dashed', color='red')
plt.title("Erro ao longo do treinamento")
plt.xlabel("Época")
plt.ylabel("Erro")
plt.legend()
plt.show()
```

**Análise da Comparação:**
- A ReLU geralmente converge mais rapidamente devido à sua derivada constante (1) para entradas positivas
- A Sigmoid pode ser mais estável para certos problemas devido à sua natureza limitada
- A escolha entre elas depende do problema específico e da arquitetura da rede

## 🧪 Teste do Modelo

Após o treinamento, o código testa os modelos com os primeiros 10 pontos do dataset:

```python
print("\nTeste do modelo com Sigmoid:")
for i in range(10):
    weighted_sum = np.dot(X[i], weights_sigmoid) + bias_sigmoid
    output = sigmoid(weighted_sum)
    print(f"Entrada: {X[i]}, Saída esperada: {y[i]}, Saída obtida: {output}")

print("\nTeste do modelo com ReLU:")
for i in range(10):
    weighted_sum = np.dot(X[i], weights_relu) + bias_relu
    output = relu(weighted_sum)
    print(f"Entrada: {X[i]}, Saída esperada: {y[i]}, Saída obtida: {output}")
```

Para cada ponto de teste:
1. Calcula a soma ponderada usando os pesos e viés treinados
2. Aplica a função de ativação correspondente
3. Compara a saída obtida com o valor esperado

**Interpretação dos Resultados:**
- Para a Sigmoid, valores próximos de 1 indicam pontos dentro do círculo, e valores próximos de 0 indicam pontos fora
- Para a ReLU, valores positivos geralmente indicam pontos dentro do círculo, e valores próximos de 0 indicam pontos fora
- A precisão do modelo pode ser avaliada pela proximidade entre a saída obtida e o valor esperado

## 🔑 Conceitos-Chave

1. **Perceptron**: A rede implementada é um perceptron simples (rede neural de camada única)
2. **Gradiente Descendente**: O algoritmo de otimização usado para minimizar o erro
3. **Funções de Ativação**: Introduzem não-linearidade, permitindo que a rede aprenda padrões complexos
4. **Backpropagation**: O processo de propagar o erro de volta pela rede para ajustar os pesos
5. **Erro Quadrático Médio**: A função de custo que mede a performance do modelo

Este código demonstra os fundamentos do aprendizado de máquina e redes neurais em um problema de classificação simples mas visualmente intuitivo.
