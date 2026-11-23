#include <MathUtils.h>

#include <stdexcept>
#include <string>
#include <algorithm>
namespace dnn {
    double sum(const std::vector<double>& values) {
        double sum = 0.0;
        for (const double& value : values) {
            sum += value;
        }

        return sum;
    }

    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (int i = 0; i < a.size(); i++) {
            sum += a[i] * b[i];
        }

        return sum;
    }
    std::vector<double> encodeAsOneHot(double value, std::size_t numClasses) {
        if (value < 0.0 || value >= static_cast<double>(numClasses)) {
            throw std::out_of_range("encodeAsOneHot: value " + std::to_string(value) + " is out of range for " +
                                    std::to_string(numClasses) + " classes");
        }

        std::vector<double> encoded(numClasses, 0.0);
        encoded[static_cast<std::size_t>(value)] = 1.0;
        return encoded;
    }

}
