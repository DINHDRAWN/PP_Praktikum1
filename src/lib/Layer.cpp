#include <Layer.h>

#include <MathUtils.h>

#include <cmath>
#include <iostream>
#include <utility>

namespace dnn {

Layer::Layer(uint64_t id, uint64_t seed, uint64_t layerInputSize, uint64_t layerOutputSize,
             ActivationFunction* activationFunction, std::string  name)
    : id(id), name(std::move(name)), seed(seed), layerInputSize(layerInputSize), layerOutputSize(layerOutputSize),
      activationFunction(activationFunction) {
    accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    accumulatedDBias.assign(layerOutputSize, 0.0);
}

Layer::~Layer() {
    delete activationFunction;
}

void Layer::init() {
    std::mt19937 randomGenerator(seed);

    // Glorot/Xavier uniform initialization:
    // scales the weight range by fan-in/fan-out so pre-activation variance stays roughly O(1)
    // regardless of layer width, instead of growing with layerInputSize the way a fixed
    // [-1, 1] range does.
    const double limit = std::sqrt(6.0 / static_cast<double>(layerInputSize + layerOutputSize));
    std::uniform_real_distribution<double> distribution(-limit, limit);

    weights.assign(layerOutputSize, std::vector<double>(layerInputSize));
    for (auto& row : weights) {
        for (double& value : row) {
            value = distribution(randomGenerator);
        }
    }

    // Zero-initialized bias
    bias.assign(layerOutputSize, 0.0);
}

void Layer::setParameters(std::vector<std::vector<double>> weights, std::vector<double> bias) {
    this->weights = std::move(weights);
    this->bias = std::move(bias);
}

uint64_t Layer::getId() const {
    return id;
}

const std::string& Layer::getName() const {
    return name;
}

const std::vector<double>& Layer::getActivationOutput() const {
    return activationOutput;
}

const std::vector<double>& Layer::getPreActivationOutput() const {
    return preActivationOutput;
}

ActivationFunction* Layer::getActivationFunction() const {
    return activationFunction;
}

const std::vector<std::vector<double>>& Layer::getWeights() const {
    return weights;
}

const std::vector<double>& Layer::getBias() const {
    return bias;
}

const std::vector<double>& Layer::getDPreviousLayerOutput() const {
    return dPreviousLayerOutput;
}

const std::vector<std::vector<double>>& Layer::getDWeights() const {
    return dWeights;
}

const std::vector<double>& Layer::getDBias() const {
    return dBias;
}
//implemented forward Propagation
void Layer::forwardPropagation(const std::vector<double>& previousLayerOutput) {
    preActivationOutput.assign(layerOutputSize, 0.0);
    activationOutput.assign(layerOutputSize, 0.0);

    for (int i = 0; i < layerOutputSize; i++) {
        preActivationOutput[i] = dnn::dotProduct(weights[i], previousLayerOutput) + bias[i];
        activationOutput[i] = getActivationFunction()-> activation(preActivationOutput[i]);
    }
}

//implemented backward Propagation
void Layer::backwardPropagation(const std::vector<double>& dActivationOutput, const std::vector<double>& previousLayerOutput) {
    if (getActivationFunction() == nullptr) {
        throw std::logic_error("No activation function");
    }
    if (dActivationOutput.size() == 0 || previousLayerOutput.size() == 0) {
        throw std::invalid_argument("Input vector dimensions do not match layer dimensions.");
    }
    //1 preactivation
    std::vector<double> dPreActivation(layerOutputSize, 0.0);
    for (int i = 0; i < layerOutputSize; i++) {
        dPreActivation[i] = dActivationOutput[i] * getActivationFunction()->activationBackwards(preActivationOutput[i]);
    }
    //2
    dWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));

    for (int i = 0; i < layerOutputSize; i++) {
        for (int j = 0; j < layerInputSize; j++) {
            dWeights[i][j] = dPreActivation[i] * previousLayerOutput[j];
        }
    }
    //save dActivation in dBias
    this->dBias.assign(dPreActivation.begin(), dPreActivation.end());
    //accumulated Weights update
    for (int i = 0; i < layerOutputSize; i++) {
        for (int j = 0; j < layerInputSize; j++) {
            accumulatedDWeights[i][j] = accumulatedDWeights[i][j] + dWeights[i][j];
        }
    }
    //accumulated Bias update
    for (int i = 0; i < dPreActivation.size(); i++) {
        accumulatedDBias[i] = accumulatedDBias[i] + dBias[i];
    }
    //increment number of processed samples
    numProcessedSamples++;

    //
    dPreviousLayerOutput.assign(layerInputSize, 0.0);
    for (int j = 0; j < layerInputSize; j++) {
        std::vector<double> weightsColumn(layerOutputSize, 0.0);
        for (int i = 0; i < weightsColumn.size(); i++) {
            weightsColumn[i] = weights[i][j];
        }
        dPreviousLayerOutput[j] = dotProduct(weightsColumn, dPreActivation);
    }

}
void Layer::updateParameters(const std::vector<std::vector<double>>& dWeights, const std::vector<double>& dBias, double learningRate) {
    for (int i = 0; i < dWeights.size(); i++) {
        for (int j = 0; j < dWeights[i].size(); j++) {
            weights[i][j] = weights[i][j] - learningRate * dWeights[i][j];
        }
    }

    for (int i = 0; i < dBias.size(); i++) {
            bias[i] = bias[i] - learningRate * dBias[i];
        }
}
void Layer::commitToWeights(double learningRate) {
    if (numProcessedSamples == 0) {
        return;
    }
    const double numSamples = static_cast<double>(numProcessedSamples);
    for (int i = 0; i < accumulatedDWeights.size(); i++) {
        for (int j = 0; j < accumulatedDWeights[i].size(); j++) {
            accumulatedDWeights[i][j] = accumulatedDWeights[i][j] / numSamples;
        }
    }

    for (int i = 0; i < accumulatedDBias.size(); i++) {
        accumulatedDBias[i] = accumulatedDBias[i] / numSamples;
    }
    //update parameters
    updateParameters(accumulatedDWeights, accumulatedDBias, learningRate);
    //reset weights and bias
    for (int i = 0; i < accumulatedDWeights.size(); i++) {
        for (int j = 0; j < accumulatedDWeights[i].size(); j++) {
            accumulatedDWeights[i][j] = 0.0;
        }
    }

    for (int i = 0; i < accumulatedDBias.size(); i++) {
        accumulatedDBias[i] = 0.0;
    }
    //reset number of processed samples
    numProcessedSamples = 0;
}

}
