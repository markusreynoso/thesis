import numpy as np
import pandas as pd


def sig(x: np.array):
    return 1/(1 + np.exp(-x))


def sigPrime(x: np.array):
    return sig(x) * (1-sig(x))


def softmax(x: np.array):
    # Clipping for numerical stability
    x = np.clip(x, -500, 500)
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)


class Layer:
    def __init__(self, inputSize: int, outputSize: int, activation="sigmoid"):
        self.size = outputSize
        self.weights = np.random.randn(outputSize, inputSize)
        self.biases = np.random.randn(outputSize, 1)
        self.z = np.zeros(shape=(outputSize, 1))
        self.activation = activation

        # ADAM
        self.t = 1
        self.mWeights = np.zeros(shape=(outputSize, 1))
        self.vWeights = np.zeros(shape=(outputSize, 1))
        self.mBiases = np.zeros(shape=(outputSize, 1))
        self.vBiases = np.zeros(shape=(outputSize, 1))

    def forward(self, inputVector: np.array):
        self.z = (self.weights @ inputVector) + self.biases
        return self.act(self.z)

    def act(self, x: np.array):
        if self.activation == "sigmoid":
            return sig(x)
        elif self.activation == "relu":
            return np.where(x < 0, 0.0, x)
        elif self.activation == "softmax":
            return softmax(x)

    def actPrime(self, x: np.array):
        if self.activation == "sigmoid":
            return sigPrime(x)
        elif self.activation == "relu":
            return np.where(x > 0, 1.0, 0.0)

    def resetLayerAdam(self):
        self.t = 1
        self.mWeights = np.zeros(shape=(self.size, 1))
        self.vWeights = np.zeros(shape=(self.size, 1))


class MLP:
    def __init__(self, hiddenLayersInput: list[Layer], learnRate: float, lossFunction="mse"):
        self.hiddenLayers: list[Layer] = []
        for i in range(len(hiddenLayersInput) - 1):
            self.hiddenLayers.append(hiddenLayersInput[i])
        self.outputLayer = hiddenLayersInput[-1]
        self.learnRate = learnRate
        self.isTrained = False
        self.lossFunction = lossFunction

    def forwardPass(self, inputVector: np.array):
        outputVector = inputVector
        for layer in self.hiddenLayers:
            outputVector = layer.forward(outputVector)

        return self.outputLayer.forward(outputVector)

    def sgd(self, x, yActual, regularizers: dict = None):
        if regularizers is None:
            regularizers = {'l1': 0, 'l2': 0, 'l0Beta': 0, 'l0Lambda': 0}

        l0Lambda = regularizers.get('l0Lambda', 0)
        l0Beta = regularizers.get('l0Beta', 0)
        l1Lambda = regularizers.get('l1', 0)
        l2Lambda = regularizers.get('l2', 0)

        yPred = self.forwardPass(x)
        gradient = None

        if self.lossFunction != "ce":
            gradient = yPred - yActual
            # δ^L = ∇_(a^L)C ⊙ a'^(L)
            delta = gradient * self.outputLayer.actPrime(self.outputLayer.z)
        else:
            delta = yPred - yActual

        # w^L ← w^L - n(δ^L)(a^(L-1))^T
        self.outputLayer.weights -= (self.learnRate *
                                     ((delta @ self.hiddenLayers[-1].act(self.hiddenLayers[-1].z).T)
                                      + (l2Lambda * self.outputLayer.weights)
                                      + (l1Lambda * np.sign(self.outputLayer.weights))
                                      + (l0Lambda * l0Beta * np.sign(self.outputLayer.weights) * np.exp(np.clip(-l0Beta * np.abs(self.outputLayer.weights), -10, 10)))
                                      )
                                     )

        # b^{L}_j ← b^{L}_j - β(δ^{L}_j)
        self.outputLayer.biases -= self.learnRate * delta

        for layer in range(len(self.hiddenLayers) - 1, -1, -1):
            # δ^l = a'^(l) ⊙ ((w^(l+1))^T δ^(l+1))
            if layer == len(self.hiddenLayers) - 1:
                delta = self.hiddenLayers[layer].actPrime(self.hiddenLayers[layer].z) * (self.outputLayer.weights.T @ delta)
            else:
                delta = self.hiddenLayers[layer].actPrime(self.hiddenLayers[layer].z) * (self.hiddenLayers[layer+1].weights.T @ delta)

            # w^l ← w^l - β(δ^l)[a^(l-1)]^T
            if layer != 0:
                self.hiddenLayers[layer].weights -= (self.learnRate *
                                                     ((delta @ self.hiddenLayers[layer - 1].act(self.hiddenLayers[layer - 1].z).T)
                                                      + (l2Lambda * self.hiddenLayers[layer].weights)
                                                      + (l1Lambda * np.sign(self.hiddenLayers[layer].weights))
                                                      + (l0Lambda * l0Beta * np.sign(self.hiddenLayers[layer].weights) * np.exp(np.clip(-l0Beta * np.abs(self.hiddenLayers[layer].weights), -10, 10)))
                                                      )
                                                     )
            else:
                self.hiddenLayers[layer].weights -= (self.learnRate *
                                                     ((delta @ x.T)
                                                      + (l2Lambda * self.hiddenLayers[layer].weights)
                                                      + (l1Lambda * np.sign(self.hiddenLayers[layer].weights))
                                                      + (l0Lambda * l0Beta * np.sign(self.hiddenLayers[layer].weights) * np.exp(np.clip(-l0Beta * np.abs(self.hiddenLayers[layer].weights), -10, 10)))
                                                      )
                                                     )
            # b^{l}_j ← b^{l}_j - β(δ^{l}_j)
            self.hiddenLayers[layer].biases -= self.learnRate * delta

    def adam(self, x, yActual, regularizers: dict = None, beta1: float = 0.9, beta2: float = 0.99):
        if regularizers is None:
            regularizers = {'l1': 0, 'l2': 0, 'l0Beta': 0, 'l0Lambda': 0}

        l0Lambda = regularizers.get('l0Lambda', 0)
        l0Beta = regularizers.get('l0Beta', 0)
        l1Lambda = regularizers.get('l1', 0)
        l2Lambda = regularizers.get('l2', 0)

        yPred = self.forwardPass(x)
        gradient = None

        if self.lossFunction != "ce":
            gradient = yPred - yActual
            # δ^L = ∇_(a^L)C ⊙ a'^(L)
            delta = gradient * self.outputLayer.actPrime(self.outputLayer.z)
        else:
            delta = yPred - yActual

        # w^L ← w^L - β(δ^L)(a^(L-1))^T
        wMoment1 = beta1 * self.outputLayer.mWeights + (1 - beta1) * delta
        wMoment2 = beta2 * self.outputLayer.vWeights + (1 - beta2) * (delta ** 2)
        bMoment1 = beta1 * self.outputLayer.mBiases + (1 - beta1) * delta
        bMoment2 = beta2 * self.outputLayer.vBiases + (1 - beta2) * (delta ** 2)

        wMoment1Adj = wMoment1 / (1 - beta1**self.outputLayer.t)
        wMoment2Adj = wMoment2 / (1 - beta2**self.outputLayer.t)
        bMoment1Adj = bMoment1 / (1 - beta1**self.outputLayer.t)
        bMoment2Adj = bMoment2 / (1 - beta2**self.outputLayer.t)

        wAdamCoeff = wMoment1Adj / (np.sqrt(wMoment2Adj) + 0.00000001)
        bAdamCoeff = bMoment1Adj / (np.sqrt(bMoment2Adj) + 0.00000001)

        self.outputLayer.mWeights = wMoment1
        self.outputLayer.vWeights = wMoment2
        self.outputLayer.mBiases = bMoment1
        self.outputLayer.vBiases = bMoment2
        self.outputLayer.t += 1

        self.outputLayer.weights -= (self.learnRate *
                                     ((wAdamCoeff @ self.hiddenLayers[-1].act(self.hiddenLayers[-1].z).T)
                                      + (l2Lambda * self.outputLayer.weights)
                                      + (l1Lambda * np.sign(self.outputLayer.weights))
                                      + (l0Lambda * l0Beta * np.sign(self.outputLayer.weights) * np.exp(np.clip(-l0Beta * np.abs(self.outputLayer.weights), -10, 10)))
                                      )
                                     )

        # b^{L}_j ← b^{L}_j - β(δ^{L}_j)
        self.outputLayer.biases -= self.learnRate * bAdamCoeff

        for layer in range(len(self.hiddenLayers) - 1, -1, -1):
            # δ^l = a'^(l) ⊙ ((w^(l+1))^T δ^(l+1))
            if layer == len(self.hiddenLayers) - 1:
                delta = self.hiddenLayers[layer].actPrime(self.hiddenLayers[layer].z) * (self.outputLayer.weights.T @ delta)
            else:
                delta = self.hiddenLayers[layer].actPrime(self.hiddenLayers[layer].z) * (self.hiddenLayers[layer+1].weights.T @ delta)

            # w^l ← w^l - β(δ^l)[a^(l-1)]^T
            if layer != 0:
                wMoment1 = beta1 * self.hiddenLayers[layer].mWeights + (1 - beta1) * delta
                wMoment2 = beta2 * self.hiddenLayers[layer].vWeights + (1 - beta2) * (delta ** 2)
                bMoment1 = beta1 * self.hiddenLayers[layer].mBiases + (1 - beta1) * delta
                bMoment2 = beta2 * self.hiddenLayers[layer].vBiases + (1 - beta2) * (delta ** 2)

                wMoment1Adj = wMoment1 / (1 - beta1**self.hiddenLayers[layer].t)
                wMoment2Adj = wMoment2 / (1 - beta2**self.hiddenLayers[layer].t)
                bMoment1Adj = bMoment1 / (1 - beta1**self.hiddenLayers[layer].t)
                bMoment2Adj = bMoment2 / (1 - beta2**self.hiddenLayers[layer].t)

                wAdamCoeff = wMoment1Adj / (np.sqrt(wMoment2Adj) + 0.00000001)
                bAdamCoeff = bMoment1Adj / (np.sqrt(bMoment2Adj) + 0.00000001)

                self.hiddenLayers[layer].mWeights = wMoment1
                self.hiddenLayers[layer].vWeights = wMoment2
                self.hiddenLayers[layer].mBiases = bMoment1
                self.hiddenLayers[layer].vBiases = bMoment2
                self.hiddenLayers[layer].t += 1

                self.hiddenLayers[layer].weights -= (self.learnRate *
                                                     ((wAdamCoeff @ self.hiddenLayers[layer - 1].act(self.hiddenLayers[layer - 1].z).T)
                                                      + (l2Lambda * self.hiddenLayers[layer].weights)
                                                      + (l1Lambda * np.sign(self.hiddenLayers[layer].weights))
                                                      + (l0Lambda * l0Beta * np.sign(self.hiddenLayers[layer].weights) * np.exp(np.clip(-l0Beta * np.abs(self.hiddenLayers[layer].weights), -10, 10)))
                                                      )
                                                     )
            else:
                wMoment1 = beta1 * self.hiddenLayers[layer].mWeights + (1 - beta1) * delta
                wMoment2 = beta2 * self.hiddenLayers[layer].vWeights + (1 - beta2) * (delta ** 2)
                bMoment1 = beta1 * self.hiddenLayers[layer].mBiases + (1 - beta1) * delta
                bMoment2 = beta2 * self.hiddenLayers[layer].vBiases + (1 - beta2) * (delta ** 2)

                wMoment1Adj = wMoment1 / (1 - beta1**self.hiddenLayers[layer].t)
                wMoment2Adj = wMoment2 / (1 - beta2**self.hiddenLayers[layer].t)
                bMoment1Adj = bMoment1 / (1 - beta1**self.hiddenLayers[layer].t)
                bMoment2Adj = bMoment2 / (1 - beta2**self.hiddenLayers[layer].t)

                wAdamCoeff = wMoment1Adj / (np.sqrt(wMoment2Adj) + 0.00000001)
                bAdamCoeff = bMoment1Adj / (np.sqrt(bMoment2Adj) + 0.00000001)

                self.hiddenLayers[layer].mWeights = wMoment1
                self.hiddenLayers[layer].vWeights = wMoment2
                self.hiddenLayers[layer].mBiases = bMoment1
                self.hiddenLayers[layer].vBiases = bMoment2
                self.hiddenLayers[layer].t += 1

                self.hiddenLayers[layer].weights -= (self.learnRate *
                                                     ((wAdamCoeff @ x.T)
                                                      + (l2Lambda * self.hiddenLayers[layer].weights)
                                                      + (l1Lambda * np.sign(self.hiddenLayers[layer].weights))
                                                      + (l0Lambda * l0Beta * np.sign(self.hiddenLayers[layer].weights) * np.exp(np.clip(-l0Beta * np.abs(self.hiddenLayers[layer].weights), -10, 10)))
                                                      )
                                                     )
            # b^{l}_j ← b^{l}_j - β(δ^{l}_j)
            self.hiddenLayers[layer].biases -= self.learnRate * bAdamCoeff

    def train(self, xTrain, yTrain, epochs: int, alg='sgd', regularizers: dict = None, beta1=0.9, beta2=0.99):
        for _ in range(epochs):
            indices = np.random.permutation(len(xTrain))
            xShuffled, yShuffled = xTrain[indices], yTrain[indices]
            for i in range(len(xTrain)):
                x = xShuffled[i].reshape(-1, 1)
                y = yShuffled[i].reshape(-1, 1)
                if alg == 'sgd':
                    self.sgd(x, y, regularizers=regularizers)
                elif alg == 'adam':
                    self.adam(x, y, regularizers=regularizers, beta1=beta1, beta2=beta2)

        if alg == 'adam':
            self.outputLayer.resetLayerAdam()
            for layer in self.hiddenLayers:
                layer.resetLayerAdam()

        self.isTrained = True

    def predict(self, x: np.array):
        if not self.isTrained:
            return -1
        else:
            return self.forwardPass(x.reshape(-1, 1))

    def test(self, xTest: np.array, yTest: np.array):
        n = len(xTest)
        correct = 0
        for i in range(n):
            prediction = self.predict(xTest[i])
            if yTest[i][np.argmax(prediction)] == 1:
                correct += 1
        return correct/n

    def confusion(self, xTest: np.array, yTest: np.array):
        n = len(xTest)
        matrix = np.zeros((2, 2))  # [ [TP, FP], [FN, TN] ]

        for i in range(n):
            prediction = self.predict(xTest[i])
            predictedClass = np.argmax(prediction)  # Predicted class
            actualClass = np.argmax(yTest[i])      # Actual class

            if actualClass == 1 and predictedClass == 1:  # True Positive (TP)
                matrix[0, 0] += 1
            elif actualClass == 0 and predictedClass == 0:  # True Negative (TN)
                matrix[1, 1] += 1
            elif actualClass == 1 and predictedClass == 0:  # False Negative (FN)
                matrix[1, 0] += 1
            elif actualClass == 0 and predictedClass == 1:  # False Positive (FP)
                matrix[0, 1] += 1

        confusion_matrix = pd.DataFrame(
            matrix,
            columns=['Predicted (+)', 'Predicted (-)'],
            index=['Actual (+)', 'Actual (-)']
        )

        precision = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1]) if (matrix[0, 0] + matrix[0, 1]) > 0 else 0
        recall = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0]) if (matrix[0, 0] + matrix[1, 0]) > 0 else 0

        print(confusion_matrix)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    def pruneNode(self, layerIdx, nodeIdx):
        self.hiddenLayers[layerIdx].weights = np.delete(self.hiddenLayers[layerIdx].weights, nodeIdx, axis=0)
        self.hiddenLayers[layerIdx].biases = np.delete(self.hiddenLayers[layerIdx].biases, nodeIdx, axis=0)
        if layerIdx == len(self.hiddenLayers)-1:
            self.outputLayer.weights = np.delete(self.outputLayer.weights, nodeIdx, axis=1)
        else:
            self.hiddenLayers[layerIdx + 1].weights = np.delete(self.hiddenLayers[layerIdx + 1].weights, nodeIdx, axis=1)

    def pruneSb(self):
        minSaliency = float('inf')
        minSaliencyIdx = (0, 0, 0)

        # Locates the neuron pair with minimum saliency
        for h in range(len(self.hiddenLayers)):
            layer = self.hiddenLayers[h]
            for i in range(len(layer.weights)):
                neuronIWeights = layer.weights[i]
                for j in range(len(layer.weights)):
                    if i == j:
                        continue
                    neuronJWeights = layer.weights[j]
                    epsilon = neuronIWeights - neuronJWeights
                    epsilonNorm2 = np.inner(epsilon, epsilon)
                    if h == len(self.hiddenLayers) - 1:
                        ajSquaredMean = (self.outputLayer.weights[:, j]**2).mean()
                    else:
                        ajSquaredMean = (self.hiddenLayers[h + 1].weights[:, j]**2).mean()
                    sij = ajSquaredMean*epsilonNorm2

                    if sij < minSaliency:
                        minSaliency = sij
                        minSaliencyIdx = (h, i, j)

        layerIdx, neuronIIdx, neuronJIdx = minSaliencyIdx
        if layerIdx == len(self.hiddenLayers) - 1:
            self.outputLayer.weights[:, neuronIIdx] += self.outputLayer.weights[:, neuronJIdx]
        else:
            self.hiddenLayers[layerIdx + 1].weights[:, neuronIIdx] += self.hiddenLayers[layerIdx + 1].weights[:, neuronJIdx]
        self.pruneNode(layerIdx, neuronJIdx)

    def pruneSmall(self, tolerance):
        for layer in self.hiddenLayers:
            layer.weights[np.abs(layer.weights) < tolerance] = 0

        self.outputLayer.weights[np.abs(self.outputLayer.weights) < tolerance] = 0


