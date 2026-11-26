#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "Tensor.hpp"
#include "Exceptions.hpp"

class TensorNode;
using TensorNodePtr = std::shared_ptr<TensorNode>;

enum class OpType {
    None,
    Add,
    Sub,
    Mul,
    MatMul,
    Transpose,
    RowBroadcastAdd,
    Sum,
    Mean,
    ReLU,
    Sigmoid,
    Tanh,
    MSELoss,
    SoftmaxCrossEntropyLoss
};

class TensorNode : public std::enable_shared_from_this<TensorNode> {
public:
    Tensor2D data;
    Tensor2D grad;
    bool requires_grad;
    std::vector<TensorNodePtr> children;
    OpType op;
    std::function<void()> backward_fn;

    explicit TensorNode(Tensor2D d, bool req_grad = true, OpType operation = OpType::None, std::vector<TensorNodePtr> kids = {})
        : data(std::move(d)),
          grad(data.rows(), data.cols(), 0.0),
          requires_grad(req_grad),
          children(std::move(kids)),
          op(operation),
          backward_fn([](){}) {}

    static TensorNodePtr create(Tensor2D d, bool req_grad = true, OpType operation = OpType::None, std::vector<TensorNodePtr> kids = {}) {
        return std::make_shared<TensorNode>(std::move(d), req_grad, operation, std::move(kids));
    }

    void zeroGrad() {
        grad.fill(0.0);
    }

    void backward() {
        std::vector<TensorNodePtr> topo;
        std::unordered_set<TensorNode*> visited;
        std::function<void(const TensorNodePtr&)> build_topo = [&](const TensorNodePtr& v) {
            if (!v) return;
            if (visited.find(v.get()) == visited.end()) {
                visited.insert(v.get());
                for (const auto& child : v->children) {
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };
        build_topo(shared_from_this());

        if (grad.rows() != data.rows() || grad.cols() != data.cols()) {
            grad = Tensor2D(data.rows(), data.cols(), 0.0);
        }
        grad.fill(1.0);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_fn) {
                (*it)->backward_fn();
            }
        }
    }
};

inline void accumulateGrad(Tensor2D& target_grad, const Tensor2D& incoming_grad) {
    if (target_grad.rows() == incoming_grad.rows() && target_grad.cols() == incoming_grad.cols()) {
        const size_t total = target_grad.rows() * target_grad.cols();
        const double* src = incoming_grad.data().data();
        double* dst = target_grad.data().data();
        for (size_t i = 0; i < total; ++i) {
            dst[i] += src[i];
        }
    } else if (incoming_grad.rows() == 1 && target_grad.cols() == incoming_grad.cols()) {
        size_t t_rows = target_grad.rows();
        size_t t_cols = target_grad.cols();
        for (size_t i = 0; i < t_rows; ++i) {
            for (size_t j = 0; j < t_cols; ++j) {
                target_grad(i, j) += incoming_grad(0, j);
            }
        }
    } else if (incoming_grad.cols() == 1 && target_grad.rows() == incoming_grad.rows()) {
        size_t t_rows = target_grad.rows();
        size_t t_cols = target_grad.cols();
        for (size_t i = 0; i < t_rows; ++i) {
            for (size_t j = 0; j < t_cols; ++j) {
                target_grad(i, j) += incoming_grad(i, 0);
            }
        }
    } else {
        throw minigrad::ShapeMismatchException("Incompatible shapes for gradient accumulation with broadcasting.");
    }
}

inline TensorNodePtr add(const TensorNodePtr& a, const TensorNodePtr& b) {
    if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols()) {
        throw minigrad::ShapeMismatchException("Shape mismatch in Autograd add operation.");
    }
    Tensor2D out_data = a->data + b->data;
    bool req = a->requires_grad || b->requires_grad;
    auto out = TensorNode::create(out_data, req, OpType::Add, {a, b});

    out->backward_fn = [out, a, b]() {
        if (a->requires_grad) {
            accumulateGrad(a->grad, out->grad);
        }
        if (b->requires_grad) {
            accumulateGrad(b->grad, out->grad);
        }
    };
    return out;
}

inline TensorNodePtr sub(const TensorNodePtr& a, const TensorNodePtr& b) {
    if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols()) {
        throw minigrad::ShapeMismatchException("Shape mismatch in Autograd sub operation.");
    }
    Tensor2D out_data = a->data - b->data;
    bool req = a->requires_grad || b->requires_grad;
    auto out = TensorNode::create(out_data, req, OpType::Sub, {a, b});

    out->backward_fn = [out, a, b]() {
        if (a->requires_grad) {
            accumulateGrad(a->grad, out->grad);
        }
        if (b->requires_grad) {
            Tensor2D neg_grad = out->grad * (-1.0);
            accumulateGrad(b->grad, neg_grad);
        }
    };
    return out;
}

inline TensorNodePtr mul(const TensorNodePtr& a, const TensorNodePtr& b) {
    if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols()) {
        throw minigrad::ShapeMismatchException("Shape mismatch in Autograd element-wise mul operation.");
    }
    Tensor2D out_data = a->data * b->data;
    bool req = a->requires_grad || b->requires_grad;
    auto out = TensorNode::create(out_data, req, OpType::Mul, {a, b});

    out->backward_fn = [out, a, b]() {
        if (a->requires_grad) {
            Tensor2D grad_a = out->grad * b->data;
            accumulateGrad(a->grad, grad_a);
        }
        if (b->requires_grad) {
            Tensor2D grad_b = out->grad * a->data;
            accumulateGrad(b->grad, grad_b);
        }
    };
    return out;
}

inline TensorNodePtr matmul(const TensorNodePtr& a, const TensorNodePtr& b) {
    if (a->data.cols() != b->data.rows()) {
        throw minigrad::ShapeMismatchException("Shape mismatch in Autograd matmul operation.");
    }
    Tensor2D out_data = a->data.matmul(b->data);
    bool req = a->requires_grad || b->requires_grad;
    auto out = TensorNode::create(out_data, req, OpType::MatMul, {a, b});

    out->backward_fn = [out, a, b]() {
        if (a->requires_grad) {
            Tensor2D grad_a = out->grad.matmul(b->data.transpose());
            accumulateGrad(a->grad, grad_a);
        }
        if (b->requires_grad) {
            Tensor2D grad_b = a->data.transpose().matmul(out->grad);
            accumulateGrad(b->grad, grad_b);
        }
    };
    return out;
}

inline TensorNodePtr transpose(const TensorNodePtr& a) {
    Tensor2D out_data = a->data.transpose();
    auto out = TensorNode::create(out_data, a->requires_grad, OpType::Transpose, {a});

    out->backward_fn = [out, a]() {
        if (a->requires_grad) {
            Tensor2D grad_a = out->grad.transpose();
            accumulateGrad(a->grad, grad_a);
        }
    };
    return out;
}

inline TensorNodePtr rowBroadcastAdd(const TensorNodePtr& a, const TensorNodePtr& bias) {
    if (bias->data.rows() != 1 || bias->data.cols() != a->data.cols()) {
        throw minigrad::ShapeMismatchException("Shape mismatch in rowBroadcastAdd: bias must be shape (1, cols).");
    }
    size_t rows = a->data.rows();
    size_t cols = a->data.cols();
    Tensor2D out_data(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out_data(i, j) = a->data(i, j) + bias->data(0, j);
        }
    }
    bool req = a->requires_grad || bias->requires_grad;
    auto out = TensorNode::create(out_data, req, OpType::RowBroadcastAdd, {a, bias});

    out->backward_fn = [out, a, bias]() {
        if (a->requires_grad) {
            accumulateGrad(a->grad, out->grad);
        }
        if (bias->requires_grad) {
            size_t r = out->grad.rows();
            size_t c = out->grad.cols();
            Tensor2D bias_grad(1, c, 0.0);
            for (size_t j = 0; j < c; ++j) {
                double sum_val = 0.0;
                for (size_t i = 0; i < r; ++i) {
                    sum_val += out->grad(i, j);
                }
                bias_grad(0, j) = sum_val;
            }
            accumulateGrad(bias->grad, bias_grad);
        }
    };
    return out;
}

inline TensorNodePtr sum(const TensorNodePtr& a) {
    Tensor2D out_data(1, 1, a->data.sum());
    auto out = TensorNode::create(out_data, a->requires_grad, OpType::Sum, {a});

    out->backward_fn = [out, a]() {
        if (a->requires_grad) {
            double g = out->grad(0, 0);
            Tensor2D grad_a(a->data.rows(), a->data.cols(), g);
            accumulateGrad(a->grad, grad_a);
        }
    };
    return out;
}

inline TensorNodePtr mean(const TensorNodePtr& a) {
    double m_val = a->data.mean();
    Tensor2D out_data(1, 1, m_val);
    auto out = TensorNode::create(out_data, a->requires_grad, OpType::Mean, {a});

    out->backward_fn = [out, a]() {
        if (a->requires_grad) {
            double scale = 1.0 / static_cast<double>(a->data.rows() * a->data.cols());
            double g = out->grad(0, 0) * scale;
            Tensor2D grad_a(a->data.rows(), a->data.cols(), g);
            accumulateGrad(a->grad, grad_a);
        }
    };
    return out;
}

inline TensorNodePtr relu(const TensorNodePtr& a) {
    size_t rows = a->data.rows();
    size_t cols = a->data.cols();
    Tensor2D out_data(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double val = a->data(i, j);
            out_data(i, j) = val > 0.0 ? val : 0.0;
        }
    }
    auto out = TensorNode::create(out_data, a->requires_grad, OpType::ReLU, {a});

    out->backward_fn = [out, a]() {
        if (a->requires_grad) {
            size_t r = a->data.rows();
            size_t c = a->data.cols();
            Tensor2D grad_a(r, c, 0.0);
            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    if (a->data(i, j) > 0.0) {
                        grad_a(i, j) = out->grad(i, j);
                    }
                }
            }
            accumulateGrad(a->grad, grad_a);
        }
    };
    return out;
}

inline TensorNodePtr sigmoid(const TensorNodePtr& a) {
    size_t rows = a->data.rows();
    size_t cols = a->data.cols();
    Tensor2D out_data(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double val = a->data(i, j);
            out_data(i, j) = 1.0 / (1.0 + std::exp(-val));
        }
    }
    auto out = TensorNode::create(out_data, a->requires_grad, OpType::Sigmoid, {a});

    out->backward_fn = [out, a]() {
        if (a->requires_grad) {
            size_t r = a->data.rows();
            size_t c = a->data.cols();
            Tensor2D grad_a(r, c, 0.0);
            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    double s = out->data(i, j);
                    grad_a(i, j) = out->grad(i, j) * s * (1.0 - s);
                }
            }
            accumulateGrad(a->grad, grad_a);
        }
    };
    return out;
}

inline TensorNodePtr tanh_op(const TensorNodePtr& a) {
    size_t rows = a->data.rows();
    size_t cols = a->data.cols();
    Tensor2D out_data(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out_data(i, j) = std::tanh(a->data(i, j));
        }
    }
    auto out = TensorNode::create(out_data, a->requires_grad, OpType::Tanh, {a});

    out->backward_fn = [out, a]() {
        if (a->requires_grad) {
            size_t r = a->data.rows();
            size_t c = a->data.cols();
            Tensor2D grad_a(r, c, 0.0);
            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    double t = out->data(i, j);
                    grad_a(i, j) = out->grad(i, j) * (1.0 - t * t);
                }
            }
            accumulateGrad(a->grad, grad_a);
        }
    };
    return out;
}

inline TensorNodePtr mseLoss(const TensorNodePtr& pred, const TensorNodePtr& target) {
    if (pred->data.rows() != target->data.rows() || pred->data.cols() != target->data.cols()) {
        throw minigrad::ShapeMismatchException("Shape mismatch in mseLoss.");
    }
    size_t rows = pred->data.rows();
    size_t cols = pred->data.cols();
    double total_sum = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double diff = pred->data(i, j) - target->data(i, j);
            total_sum += diff * diff;
        }
    }
    double n = static_cast<double>(rows * cols);
    double loss_val = total_sum / n;
    Tensor2D out_data(1, 1, loss_val);
    bool req = pred->requires_grad || target->requires_grad;
    auto out = TensorNode::create(out_data, req, OpType::MSELoss, {pred, target});

    out->backward_fn = [out, pred, target]() {
        if (pred->requires_grad) {
            size_t r = pred->data.rows();
            size_t c = pred->data.cols();
            double scale = 2.0 / static_cast<double>(r * c);
            double g_scalar = out->grad(0, 0);
            Tensor2D grad_pred(r, c);
            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    grad_pred(i, j) = g_scalar * scale * (pred->data(i, j) - target->data(i, j));
                }
            }
            accumulateGrad(pred->grad, grad_pred);
        }
        if (target->requires_grad) {
            size_t r = target->data.rows();
            size_t c = target->data.cols();
            double scale = 2.0 / static_cast<double>(r * c);
            double g_scalar = out->grad(0, 0);
            Tensor2D grad_target(r, c);
            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    grad_target(i, j) = g_scalar * scale * (target->data(i, j) - pred->data(i, j));
                }
            }
            accumulateGrad(target->grad, grad_target);
        }
    };
    return out;
}

inline TensorNodePtr softmaxCrossEntropyLoss(const TensorNodePtr& logits, const TensorNodePtr& targets) {
    if (logits->data.rows() != targets->data.rows() || logits->data.cols() != targets->data.cols()) {
        throw minigrad::ShapeMismatchException("Shape mismatch in softmaxCrossEntropyLoss.");
    }
    size_t rows = logits->data.rows();
    size_t cols = logits->data.cols();

    Tensor2D probs(rows, cols);
    double total_loss = 0.0;

    for (size_t i = 0; i < rows; ++i) {
        double max_val = logits->data(i, 0);
        for (size_t j = 1; j < cols; ++j) {
            if (logits->data(i, j) > max_val) {
                max_val = logits->data(i, j);
            }
        }
        double sum_exp = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            sum_exp += std::exp(logits->data(i, j) - max_val);
        }
        for (size_t j = 0; j < cols; ++j) {
            double p = std::exp(logits->data(i, j) - max_val) / sum_exp;
            probs(i, j) = p;
            double t = targets->data(i, j);
            if (t > 0.0) {
                total_loss -= t * std::log(p + 1e-15);
            }
        }
    }

    double mean_loss = total_loss / static_cast<double>(rows);
    Tensor2D out_data(1, 1, mean_loss);
    bool req = logits->requires_grad || targets->requires_grad;
    auto out = TensorNode::create(out_data, req, OpType::SoftmaxCrossEntropyLoss, {logits, targets});

    out->backward_fn = [out, logits, targets, rows, cols]() {
        Tensor2D p(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            double max_val = logits->data(i, 0);
            for (size_t j = 1; j < cols; ++j) {
                if (logits->data(i, j) > max_val) max_val = logits->data(i, j);
            }
            double sum_exp = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                sum_exp += std::exp(logits->data(i, j) - max_val);
            }
            for (size_t j = 0; j < cols; ++j) {
                p(i, j) = std::exp(logits->data(i, j) - max_val) / sum_exp;
            }
        }

        if (logits->requires_grad) {
            double g_scalar = out->grad(0, 0) / static_cast<double>(rows);
            Tensor2D grad_logits(rows, cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    grad_logits(i, j) = g_scalar * (p(i, j) - targets->data(i, j));
                }
            }
            accumulateGrad(logits->grad, grad_logits);
        }
        if (targets->requires_grad) {
            double g_scalar = out->grad(0, 0) / static_cast<double>(rows);
            Tensor2D grad_targets(rows, cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    grad_targets(i, j) = g_scalar * (-std::log(p(i, j) + 1e-15));
                }
            }
            accumulateGrad(targets->grad, grad_targets);
        }
    };

    return out;
}

#endif // AUTOGRAD_HPP
