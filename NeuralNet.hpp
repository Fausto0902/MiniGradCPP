#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "Tensor.hpp"
#include "Autograd.hpp"
#include "Exceptions.hpp"

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::shared_ptr<TensorNode> forward(std::shared_ptr<TensorNode> input) = 0;
    virtual std::vector<std::shared_ptr<TensorNode>> getParameters() = 0;
    virtual std::string name() const = 0;
};

class Linear : public Layer {
private:
    std::shared_ptr<TensorNode> weight_;
    std::shared_ptr<TensorNode> bias_;
    size_t in_features_;
    size_t out_features_;

public:
    Linear(size_t in_features, size_t out_features, std::mt19937& rng)
        : in_features_(in_features), out_features_(out_features) {
        
        // Xavier / Glorot Normal Initialization: stddev = sqrt(2 / (in + out))
        double stddev = std::sqrt(2.0 / static_cast<double>(in_features_ + out_features_));
        std::normal_distribution<double> dist(0.0, stddev);

        Tensor2D w_data(in_features_, out_features_);
        for (size_t i = 0; i < in_features_; ++i) {
            for (size_t j = 0; j < out_features_; ++j) {
                w_data(i, j) = dist(rng);
            }
        }

        Tensor2D b_data(1, out_features_);
        for (size_t j = 0; j < out_features_; ++j) {
            b_data(0, j) = 0.0; // Biases initialized to zero
        }

        weight_ = std::make_shared<TensorNode>(w_data, true);
        bias_ = std::make_shared<TensorNode>(b_data, true);
    }

    std::shared_ptr<TensorNode> forward(std::shared_ptr<TensorNode> input) override {
        auto mm = matmul(input, weight_);
        return rowBroadcastAdd(mm, bias_);
    }

    std::vector<std::shared_ptr<TensorNode>> getParameters() override {
        return {weight_, bias_};
    }

    std::string name() const override {
        return "Linear";
    }

    std::shared_ptr<TensorNode> getWeight() const { return weight_; }
    std::shared_ptr<TensorNode> getBias() const { return bias_; }

    const Tensor2D& getWeights() const { return weight_->data; }

    void setWeightsAndBiases(const Tensor2D& w, const Tensor2D& b) {
        if (w.rows() != in_features_ || w.cols() != out_features_) {
            throw minigrad::ShapeMismatchException("Dimensiones de peso incompatibles en Linear::setWeightsAndBiases");
        }
        if (b.rows() != 1 || b.cols() != out_features_) {
            throw minigrad::ShapeMismatchException("Dimensiones de sesgo incompatibles en Linear::setWeightsAndBiases");
        }
        weight_->data = w;
        weight_->grad = Tensor2D(w.rows(), w.cols(), 0.0);
        bias_->data = b;
        bias_->grad = Tensor2D(b.rows(), b.cols(), 0.0);
    }

    void setWeightsAndBias(const Tensor2D& w, const Tensor2D& b) {
        setWeightsAndBiases(w, b);
    }
};

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh
};

class Activation : public Layer {
private:
    ActivationType type_;

public:
    explicit Activation(ActivationType type) : type_(type) {}

    std::shared_ptr<TensorNode> forward(std::shared_ptr<TensorNode> input) override {
        switch (type_) {
            case ActivationType::ReLU:
                return relu(input);
            case ActivationType::Sigmoid:
                return sigmoid(input);
            case ActivationType::Tanh:
                return tanh_op(input);
        }
        return input;
    }

    std::vector<std::shared_ptr<TensorNode>> getParameters() override {
        return {};
    }

    std::string name() const override {
        switch (type_) {
            case ActivationType::ReLU: return "ReLU";
            case ActivationType::Sigmoid: return "Sigmoid";
            case ActivationType::Tanh: return "Tanh";
        }
        return "Activation";
    }

    ActivationType getType() const { return type_; }
};

class Sequential {
private:
    std::vector<std::shared_ptr<Layer>> layers_;

public:
    Sequential() = default;

    void add(std::shared_ptr<Layer> layer) {
        layers_.push_back(layer);
    }

    std::shared_ptr<TensorNode> forward(std::shared_ptr<TensorNode> input) {
        auto current = input;
        for (auto& layer : layers_) {
            current = layer->forward(current);
        }
        return current;
    }

    std::vector<std::shared_ptr<TensorNode>> getParameters() {
        std::vector<std::shared_ptr<TensorNode>> params;
        for (auto& layer : layers_) {
            auto p = layer->getParameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

    const std::vector<std::shared_ptr<Layer>>& getLayers() const {
        return layers_;
    }
};

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zeroGrad() = 0;
};

class SGDOptimizer : public Optimizer {
private:
    std::vector<std::shared_ptr<TensorNode>> params_;
    double lr_;
    double momentum_;
    std::vector<Tensor2D> velocities_;

public:
    SGDOptimizer(const std::vector<std::shared_ptr<TensorNode>>& params, double lr = 0.01, double momentum = 0.0)
        : params_(params), lr_(lr), momentum_(momentum) {
        for (const auto& p : params_) {
            velocities_.emplace_back(p->data.rows(), p->data.cols(), 0.0);
        }
    }

    void step() override {
        for (size_t i = 0; i < params_.size(); ++i) {
            auto& param = params_[i];
            const auto& grad = param->grad;
            auto& vel = velocities_[i];
            auto& data = param->data;

            size_t total = data.rows() * data.cols();
            for (size_t j = 0; j < total; ++j) {
                vel.data()[j] = momentum_ * vel.data()[j] + lr_ * grad.data()[j];
                data.data()[j] -= vel.data()[j];
            }
        }
    }

    void zeroGrad() override {
        for (auto& param : params_) {
            param->zeroGrad();
        }
    }
};

class AdamOptimizer : public Optimizer {
private:
    std::vector<std::shared_ptr<TensorNode>> params_;
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    int t_;
    std::vector<Tensor2D> m_;
    std::vector<Tensor2D> v_;

public:
    AdamOptimizer(const std::vector<std::shared_ptr<TensorNode>>& params, 
                  double lr = 0.001, 
                  double beta1 = 0.9, 
                  double beta2 = 0.999, 
                  double eps = 1e-8)
        : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {
        for (const auto& p : params_) {
            m_.emplace_back(p->data.rows(), p->data.cols(), 0.0);
            v_.emplace_back(p->data.rows(), p->data.cols(), 0.0);
        }
    }

    void step() override {
        t_++;
        double correction1 = 1.0 - std::pow(beta1_, static_cast<double>(t_));
        double correction2 = 1.0 - std::pow(beta2_, static_cast<double>(t_));

        for (size_t i = 0; i < params_.size(); ++i) {
            auto& param = params_[i];
            const auto& grad = param->grad;
            auto& m_vec = m_[i];
            auto& v_vec = v_[i];
            auto& data = param->data;

            size_t total = data.rows() * data.cols();
            for (size_t j = 0; j < total; ++j) {
                double g = grad.data()[j];
                
                // Update biased first moment estimate
                m_vec.data()[j] = beta1_ * m_vec.data()[j] + (1.0 - beta1_) * g;
                
                // Update biased second raw moment estimate
                v_vec.data()[j] = beta2_ * v_vec.data()[j] + (1.0 - beta2_) * g * g;

                // Compute bias-corrected first moment estimate
                double m_hat = m_vec.data()[j] / correction1;

                // Compute bias-corrected second raw moment estimate
                double v_hat = v_vec.data()[j] / correction2;

                // Update parameters
                data.data()[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }

    void zeroGrad() override {
        for (auto& param : params_) {
            param->zeroGrad();
        }
    }
};

#endif // NEURAL_NET_HPP
