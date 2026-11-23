#include "../../../include/lib/loss/BinaryCrossEntropy.h"

#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>

namespace dnn {

std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}

double BinaryCrossEntropy::getCostValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
    if (predictedOutput.size() == 0 || actualOutput.size() == 0) {
        return 0.0; // add throwable
    }
    if (predictedOutput.size() != actualOutput.size()) {
        return 0.0; // add throwable
    }

    std::vector<double> p;
    std::vector<double> q;
    std::vector<double> r;
    r.assign(predictedOutput.size(), 0.0);
    p.assign(predictedOutput.size(), 0.0);
    q.assign(predictedOutput.size(), 0.0);
    for (std::size_t i = 0; i < predictedOutput.size(); ++i) {
        p[i] = std::log(clampProbability(predictedOutput[i]));
        q[i] = std::log(1 - clampProbability(predictedOutput[i]));
        r[i] = 1 - actualOutput[i];
    }

    return -((dnn::dotProduct(actualOutput, p) + dnn::dotProduct(r, q)) / predictedOutput.size());
}

double BinaryCrossEntropy::getAccuracyValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
    if (predictedOutput.size() == 0 || actualOutput.size() == 0) {
        return 0.0; // add throwable
    }
    if (predictedOutput.size() != actualOutput.size()) {
        return 0.0; // add throwable
    }
    std::vector<double> v;
    v.assign(predictedOutput.size(), 0.0);
    std::size_t predicted_size = predictedOutput.size();
    std::size_t correct = 0;
    for (std::size_t i = 0; i < predictedOutput.size(); ++i) {
        if (predictedOutput[i] > 0.5) {
            v[i] = 1.0;
        }
        else {
            v[i] = 0.0;
        }

        if (v[i] == actualOutput[i]) {
            ++correct;
        }
    }
   
    return static_cast<double>(correct) / static_cast<double>(predicted_size);
}
std::vector<double> BinaryCrossEntropy::getGradient(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
    if (predictedOutput.size() == 0 || actualOutput.size() == 0) {
        return std::vector<double>(predictedOutput.size(), 0.0); // add throwable
    }
    if (predictedOutput.size() != actualOutput.size()) {
        return std::vector<double>(predictedOutput.size(), 0.0); // add throwable
    }
    std::vector<double> g;
    g.assign(predictedOutput.size(), 0.0);

    for (std::size_t i = 0; i < predictedOutput.size(); ++i) {
        const double numerator_left = actualOutput[i] / clampProbability(predictedOutput[i]);
        const double numerator_right = (1 - actualOutput[i]) / (1 - clampProbability(predictedOutput[i]));
        g[i] = - (numerator_left - numerator_right) / predictedOutput.size();
    }

    return g;
}

}
