#pragma once

#include <vector>
#include <algorithm>
namespace dnn {

    double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

    double sum(const std::vector<double>& values);

    // Encodes value as a one-hot vector of numClasses elements: the element at index value is 1.0
    // and every other element is 0.0. Throws std::out_of_range if value is negative or >= numClasses.
    std::vector<double> encodeAsOneHot(double value, std::size_t numClasses);

    template<typename T>
    // normalize every value in values with the greatest in values
    std::vector<double> normalize(const std::vector<T>& values) {
        std::vector<double> result;
        result.reserve(values.size()); // 1. approach: create new vector and push the new values after normalization

        double max_element = static_cast<int>(*std::max_element(values.begin(), values.end()));// find max element, include <algorithm>

        if (max_element == 0.0) {
            return std::vector<double>(values.size(), 0.0);
        }

        for (int i = 0; i < values.size(); i++) {
            result.push_back(static_cast<double>(values[i]) / max_element);
        }

        return result;
    }

    template<typename T>
    //normalize every value in values with maxValue
    std::vector<double> normalize(const std::vector<T>& values, int maxValue) {
        if (values.empty()) {
            return {};
        }

        const double divisor = static_cast<double>(maxValue);
        if (divisor == 0.0) {
            return std::vector<double>(values.size(), 0.0);
        }

        std::vector<double> result;
        result.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            result.push_back(values[i] / divisor);
        }
        return result;
    }
}
