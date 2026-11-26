#pragma once
#include <string>
#include <memory>
#include "NeuralNet.hpp"
#include "Exceptions.hpp"
namespace minigrad {
class Serializer {
public:
    static void saveModel(const Sequential& model, const std::string& filepath);
    static void loadModel(Sequential& model, const std::string& filepath);
    static void save(const std::shared_ptr<Sequential>& model, const std::string& filepath);
    static void load(const std::shared_ptr<Sequential>& model, const std::string& filepath);
};
}
