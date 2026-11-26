#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include "Exceptions.hpp"

class Tensor2D {
private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;

public:
    // Constructores y Destructor
    Tensor2D() : rows_(0), cols_(0), data_() {}

    Tensor2D(size_t rows, size_t cols, double initial_value = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, initial_value) {}

    Tensor2D(size_t rows, size_t cols, std::vector<double> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        if (data_.size() != rows_ * cols_) {
            throw minigrad::InvalidArgumentException("Tensor2D size mismatch with provided data vector.");
        }
    }

    Tensor2D(const Tensor2D&) = default;
    Tensor2D(Tensor2D&&) noexcept = default;
    Tensor2D& operator=(const Tensor2D&) = default;
    Tensor2D& operator=(Tensor2D&&) noexcept = default;
    ~Tensor2D() = default;

    // Getters
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }
    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }

    // Acceso a elementos con verificación de límites
    double& at(size_t r, size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw minigrad::InvalidArgumentException("Tensor2D index out of range: (" + std::to_string(r) + ", " + std::to_string(c) + 
                                           ") for shape (" + std::to_string(rows_) + ", " + std::to_string(cols_) + ").");
        }
        return data_[r * cols_ + c];
    }

    const double& at(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw minigrad::InvalidArgumentException("Tensor2D index out of range: (" + std::to_string(r) + ", " + std::to_string(c) + 
                                           ") for shape (" + std::to_string(rows_) + ", " + std::to_string(cols_) + ").");
        }
        return data_[r * cols_ + c];
    }

    double& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    const double& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    // Métodos de Modificación y Relleno
    void fill(double val) {
        std::fill(data_.begin(), data_.end(), val);
    }

    void reshape(size_t new_rows, size_t new_cols) {
        if (new_rows * new_cols != data_.size()) {
            throw minigrad::ShapeMismatchException("Cannot reshape Tensor2D of size " + std::to_string(data_.size()) + 
                                         " to (" + std::to_string(new_rows) + ", " + std::to_string(new_cols) + ").");
        }
        rows_ = new_rows;
        cols_ = new_cols;
    }

    // Operaciones Matriciales Avanzadas
    Tensor2D transpose() const {
        Tensor2D result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    Tensor2D matmul(const Tensor2D& other) const {
        if (cols_ != other.rows_) {
            throw minigrad::ShapeMismatchException("Matrix multiplication dimension mismatch: (" + 
                                         std::to_string(rows_) + "x" + std::to_string(cols_) + ") * (" + 
                                         std::to_string(other.rows_) + "x" + std::to_string(other.cols_) + ").");
        }
        Tensor2D result(rows_, other.cols_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                double aik = (*this)(i, k);
                for (size_t j = 0; j < other.cols_; ++j) {
                    result(i, j) += aik * other(k, j);
                }
            }
        }
        return result;
    }

    // Operaciones Elemento a Elemento
    Tensor2D operator+(const Tensor2D& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw minigrad::ShapeMismatchException("Element-wise addition dimension mismatch.");
        }
        Tensor2D result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Tensor2D operator-(const Tensor2D& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw minigrad::ShapeMismatchException("Element-wise subtraction dimension mismatch.");
        }
        Tensor2D result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    Tensor2D operator*(const Tensor2D& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw minigrad::ShapeMismatchException("Element-wise multiplication dimension mismatch.");
        }
        Tensor2D result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }

    Tensor2D operator*(double scalar) const {
        Tensor2D result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    friend Tensor2D operator*(double scalar, const Tensor2D& t) {
        return t * scalar;
    }

    Tensor2D operator/(double scalar) const {
        if (std::abs(scalar) < 1e-12) {
            throw minigrad::InvalidArgumentException("Tensor2D division by zero or near-zero scalar.");
        }
        Tensor2D result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] / scalar;
        }
        return result;
    }

    // Broadcasting por filas (ideal para sesgos (1, cols) o (rows, cols))
    Tensor2D addRowBias(const Tensor2D& bias) const {
        if (bias.cols_ != cols_ || bias.rows_ != 1) {
            throw minigrad::ShapeMismatchException("addRowBias expects bias tensor of shape (1, cols).");
        }
        Tensor2D result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) + bias(0, j);
            }
        }
        return result;
    }

    // Reducciones
    double sum() const {
        double total = 0.0;
        for (double val : data_) {
            total += val;
        }
        return total;
    }

    double mean() const {
        if (data_.empty()) return 0.0;
        return sum() / static_cast<double>(data_.size());
    }

    Tensor2D sumRows() const {
        Tensor2D result(1, cols_, 0.0);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(0, j) += (*this)(i, j);
            }
        }
        return result;
    }

    // Inicializadores deterministas Xavier / Glorot y He / Kaiming
    static Tensor2D xavier(size_t rows, size_t cols, std::mt19937& rng) {
        double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
        std::uniform_real_distribution<double> dist(-limit, limit);
        Tensor2D t(rows, cols);
        for (size_t i = 0; i < t.data_.size(); ++i) {
            t.data_[i] = dist(rng);
        }
        return t;
    }

    static Tensor2D he(size_t rows, size_t cols, std::mt19937& rng) {
        double stddev = std::sqrt(2.0 / static_cast<double>(rows));
        std::normal_distribution<double> dist(0.0, stddev);
        Tensor2D t(rows, cols);
        for (size_t i = 0; i < t.data_.size(); ++i) {
            t.data_[i] = dist(rng);
        }
        return t;
    }

    // Utilidad de depuración
    std::string toString() const {
        std::ostringstream oss;
        oss << "Tensor2D(" << rows_ << "x" << cols_ << ") [\n";
        for (size_t i = 0; i < rows_; ++i) {
            oss << "  [";
            for (size_t j = 0; j < cols_; ++j) {
                oss << std::setw(10) << std::fixed << std::setprecision(6) << (*this)(i, j);
                if (j + 1 < cols_) oss << ", ";
            }
            oss << "]\n";
        }
        oss << "]";
        return oss.str();
    }
};

#endif // TENSOR_HPP
