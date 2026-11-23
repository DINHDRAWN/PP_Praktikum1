#pragma once

#include <activation/ActivationFunction.h>

#include <string>

#include "ActivationFunction.h"
namespace dnn {
    class Sigmoid : public dnn::ActivationFunction{
    public:
        Sigmoid() = default;
        ~Sigmoid() override = default;

        double activation(double x) override;
        double activationBackwards(double x) override;
        std::string getName() const override;
    };
}
