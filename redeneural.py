import numpy as np
import matplotlib.pyplot as plt

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Gerar dataset de pontos dentro e fora de um círculo
np.random.seed(0)
X = np.random.uniform(-1, 1, (500, 2))  # 500 pontos aleatórios entre -1 e 1
y = (np.sqrt(X[:, 0]**2 + X[:, 1]**2) < 0.5).astype(int).reshape(-1, 1)  # Classe 1 se dentro do círculo

# Plot do dataset
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Spectral)
plt.title("Dataset: Pontos Dentro e Fora de um Círculo")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Classe")
plt.show()

# Hiperparâmetros
learning_rate = 0.5  # Taxa de aprendizado
epochs = 200000  # Número de épocas (quantas vezes o modelo treina nos dados)

# Treinamento do modelo
def train_neural_network(activation_function, activation_derivative):
    weights = np.random.randn(2, 1)  # Inicializa pesos aleatórios
    bias = np.random.randn()  # Inicializa o viés aleatoriamente
    errors = []

    for epoch in range(epochs):
        # Forward pass
        weighted_sum = np.dot(X, weights) + bias  # Calcula a soma ponderada
        output = activation_function(weighted_sum)  # Aplica a função de ativação

        # Cálculo do erro
        error = y - output  # Diferença entre saída esperada e saída real
        errors.append(np.mean(np.square(error)))  # Calcula erro quadrático médio

        # Backpropagation
        d_error = -2 * error / len(X)  # Derivada do erro
        d_output = activation_derivative(weighted_sum)  # Derivada da ativação
        gradient = d_error * d_output  # Gradiente para atualização

        # Atualização dos pesos e viés
        weights -= learning_rate * np.dot(X.T, gradient)
        bias -= learning_rate * np.sum(gradient)

        if epoch % 5000 == 0:
            print(f"Época {epoch}, Erro: {errors[-1]}")

    return weights, bias, errors

# Treinar redes com Sigmoid e ReLU
weights_sigmoid, bias_sigmoid, erros_sigmoid = train_neural_network(sigmoid, sigmoid_derivative)
weights_relu, bias_relu, erros_relu = train_neural_network(relu, relu_derivative)

# Plot do erro ao longo do treinamento
plt.plot(range(epochs), erros_sigmoid, label="Sigmoid", color='blue')
plt.plot(range(epochs), erros_relu, label="ReLU", linestyle='dashed', color='red')
plt.title("Erro ao longo do treinamento")
plt.xlabel("Época")
plt.ylabel("Erro")
plt.legend()
plt.show()

# Teste das redes
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

# Explicação do código:
# - dataset onde pontos são classificados como dentro ou fora de um círculo num 3d
# - funções de ativação: Sigmoid e Relu
# - Treinamento de rede neural de camada única para resolver o problema
# - Plotar a evolução do erro ao longo das épocas para comparar Sigmoid e Relu
# - No final, testamos a rede com algumas entradas do dataset e mostramos as saídas geradas.
