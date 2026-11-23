#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {
Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters,
                LossFunction*  lossFunction) : ownedLossFunction(lossFunction) {
    for (uint64_t id = 0; id < layerParameters.size(); id++) {
        const auto& [weights, bias, activationFunction] = layerParameters[id];

        //check empty matrix
        if (weights.size() == 0 || weights[0].size() == 0 || bias.size() == 0) {
            throw std::invalid_argument("Weights and bias are empty");
        }
        if (activationFunction == nullptr) {
            throw std::invalid_argument("ActivationFunction cannot be nullptr");
        }
        Layer* layer = new Layer(id,
                                nextSeed,
                                weights[0].size(),
                                weights.size(),
                                activationFunction,
                                activationFunction->getName());
        layer->setParameters(weights, bias);
        layers.push_back(layer);
    }//6a
}
Network::~Network() {
    for (Layer* layer : layers) {
        delete layer;
    }
    delete ownedLossFunction;
}//6a

void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

const LossFunction* Network::getLossFunction() const {
    return ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return output;
}

const std::vector<double>& Network::getCostHistory() const {
    return costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers = std::move(newLayers);
    initialInput.clear();
    output.clear();
    costHistory.clear();
    accuracyHistory.clear();
}
Network& Network::addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction* activationFunction) {
    if (inputDim == 0 || outputDim == 0) {
        throw std::invalid_argument("Input and output dimensions must be greater than 0");
    }
    //
    // std::vector<std::vector<double>> input(outputDim, std::vector<double>(inputDim, 0.0));
    // std::vector<double> bias(outputDim, 0.0);
    //generator
    //prng
    // const double limit = std::sqrt(6.0 / static_cast<double>(inputDim + outputDim));
    // std::mt19937 generator(nextSeed);
    // std::uniform_real_distribution<double> distribution(-limit, limit);
    // for (uint64_t outputIndex = 0; outputIndex < outputDim; outputIndex++) {
    //     for (uint64_t inputIndex = 0; inputIndex < inputDim; inputIndex++) {
    //         input[outputIndex][inputIndex] = distribution(generator);
    //     }
    // }
    uint64_t nextId = layers.size();
    Layer* layer = new Layer(nextId, nextSeed, inputDim, outputDim, activationFunction, activationFunction->getName());
    layer->init();
    nextSeed++;
    layers.push_back(layer); //6b
    return *this;
}
const std::vector<double>& Network::getLayerInput(uint64_t layerId) const {
    if (layerId == 0) {
        return initialInput;
    }
    return layers[layerId - 1]->getActivationOutput();
    //6b
}

void Network::fullForwardPropagation(const std::vector<double>& input) {
    if (input.size() == 0) {
        return;
    }
    this->initialInput = input;
    std::vector<double> nextInput(input);
    for (std::size_t layerId = 0; layerId < layers.size(); layerId++) {
        layers[layerId]->forwardPropagation(nextInput);
        nextInput = layers[layerId]->getActivationOutput();
    }
    this->output = nextInput;
}//6c

void Network::fullBackwardPropagation(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) {
    if (predictedOutput.size() == 0 || actualOutput.size() == 0) {
        return;
    }
    if (getLossFunction() == nullptr) {
        throw std::logic_error("nullptr");
    }
    std::vector<double> dACurr(getLossFunction()->getGradient(predictedOutput, actualOutput)); // calculate gradient
    for (std::size_t layerId = layers.size(); layerId-- > 0;) {
        layers[layerId]->backwardPropagation(dACurr, getLayerInput(layerId));
        dACurr = layers[layerId]->getDPreviousLayerOutput();
    }//6c
}

void Network::trainBatch(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs, std::size_t batchSize,
                     int epochs, double learningRate) {
    //check if loss function is nullptr
    if (getLossFunction() == nullptr) {
        throw std::logic_error("Loss Function ist nullptr");
    }
    if (layers.empty()) {
        throw std::logic_error("Network has zero layers.");
    }
    //check the dimensions of inputs and actualoutputs
    if (inputs.size() == 0 || actualOutputs.size() == 0 || batchSize == 0 || inputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("Possible dimension mismath, zero batch size or empty vectors");
    }
    //reset the cost and accuracy history
    costHistory.clear();
    accuracyHistory.clear();
    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        double totalCost = 0.0;
        double totalAccuracy = 0.0;

        for (std::size_t i = 0; i < inputs.size(); i++) {
            fullForwardPropagation(inputs[i]);//forward propagation of one input set

            totalCost += getLossFunction()->getCostValue(output, actualOutputs[i]);
            totalAccuracy += getLossFunction()->getAccuracyValue(output, actualOutputs[i]);

            fullBackwardPropagation(output, actualOutputs[i]);//backward propagation
            if (i % batchSize == 0 || i == inputs.size() - 1) { //after batchsize samples reset the accumulated weights and biases
                for (std::size_t layerId = 0; layerId < layers.size(); layerId++) {
                    layers[layerId]->commitToWeights(learningRate);
                }
            }
        }

        double epochCost = totalCost / static_cast<double>(inputs.size());
        double epochAccuracy = totalAccuracy / static_cast<double>(inputs.size());

        costHistory.push_back(epochCost);
        accuracyHistory.push_back(epochAccuracy);
    }
}

double Network::computeAccuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs) {
    //check if loss function is nullptr
    if (getLossFunction() == nullptr) {
        throw std::logic_error("Loss Function is nullptr");
    }
    //check the dimensions of inputs and actualoutputs
    if (inputs.size() == 0 || actualOutputs.size() == 0 || inputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("Possible dimension mismath, zero batch size or empty vectors");
    }

    double totalCorrect = 0.0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        fullForwardPropagation(inputs[i]);
        totalCorrect += getLossFunction()->getAccuracyValue(output, actualOutputs[i]);
    }
    return totalCorrect / static_cast<double>(inputs.size());
}
}
