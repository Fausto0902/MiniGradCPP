#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <cmath>
#include <cassert>
#include <fstream>
#include <cstdio>
#include "Exceptions.hpp"
#include "Tensor.hpp"
#include "Autograd.hpp"
#include "NeuralNet.hpp"
#include "Serializer.hpp"

static bool approxEqual(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) <= tol;
}

template <typename LossFn>
static void checkGrad(LossFn lossFn, const TensorNodePtr& param, double h = 1e-6, double tol = 1e-5) {
    for (size_t r = 0; r < param->data.rows(); ++r) {
        for (size_t c = 0; c < param->data.cols(); ++c) {
            double orig = param->data(r, c);
            param->data(r, c) = orig + h;
            double fp = lossFn();
            param->data(r, c) = orig - h;
            double fm = lossFn();
            param->data(r, c) = orig;
            double numGrad = (fp - fm) / (2.0 * h);
            assert(std::abs(numGrad - param->grad(r, c)) < tol);
        }
    }
}

int main() {
    // [TEST 01] Tensor2D construccion, acceso, fill y reshape
    {
        Tensor2D t(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        assert(t.rows() == 2 && t.cols() == 3 && t.size() == 6);
        assert(approxEqual(t.at(1, 2), 6.0));
        t.reshape(3, 2);
        assert(t.rows() == 3 && t.cols() == 2 && approxEqual(t(2, 0), 5.0));
        t.fill(2.5);
        assert(approxEqual(t.sum(), 15.0));
        std::cout << "[TEST 01] Tensor2D construccion y reshape -> PASSED\n";
    }

    // [TEST 02] Tensor2D matmul, transpose y aritmetica
    {
        Tensor2D a(2, 3, {1, 2, 3, 4, 5, 6});
        Tensor2D b(3, 2, {7, 8, 9, 10, 11, 12});
        Tensor2D c = a.matmul(b);
        assert(c.rows() == 2 && c.cols() == 2);
        assert(approxEqual(c(0, 0), 58.0) && approxEqual(c(1, 1), 154.0));
        Tensor2D at = a.transpose();
        assert(at.rows() == 3 && at.cols() == 2 && approxEqual(at(2, 1), 6.0));
        Tensor2D d = ((a + a) - a) * 2.0 / 2.0;
        assert(approxEqual(d(1, 2), 6.0));
        std::cout << "[TEST 02] Tensor2D matmul y aritmetica -> PASSED\n";
    }

    // [TEST 03] Tensor2D addRowBias, mean y sumRows
    {
        Tensor2D x(2, 3, {1, 2, 3, 4, 5, 6});
        Tensor2D bias(1, 3, {10, 20, 30});
        Tensor2D y = x.addRowBias(bias);
        assert(approxEqual(y(0, 0), 11.0) && approxEqual(y(1, 2), 36.0));
        assert(approxEqual(x.mean(), 3.5));
        Tensor2D sr = x.sumRows();
        assert(sr.rows() == 1 && sr.cols() == 3 && approxEqual(sr(0, 1), 7.0));
        std::cout << "[TEST 03] Tensor2D broadcasting y reducciones -> PASSED\n";
    }

    // [TEST 04] Tensor2D inicializadores deterministas Xavier y He
    {
        std::mt19937 r1(42), r2(42);
        Tensor2D x1 = Tensor2D::xavier(4, 5, r1);
        Tensor2D x2 = Tensor2D::xavier(4, 5, r2);
        assert(approxEqual(x1(2, 3), x2(2, 3), 1e-12));
        Tensor2D h1 = Tensor2D::he(4, 5, r1);
        assert(h1.rows() == 4 && h1.cols() == 5);
        std::cout << "[TEST 04] Inicializadores Xavier y He -> PASSED\n";
    }

    // [TEST 05] Excepciones tipadas en Tensor2D
    {
        bool e1 = false, e2 = false, e3 = false;
        Tensor2D a(2, 3, 1.0), b(2, 2, 1.0);
        try { (void)a.at(5, 0); } catch (const minigrad::InvalidArgumentException&) { e1 = true; }
        try { (void)a.matmul(b); } catch (const minigrad::ShapeMismatchException&) { e2 = true; }
        try { (void)(a / 0.0); } catch (const minigrad::InvalidArgumentException&) { e3 = true; }
        assert(e1 && e2 && e3);
        std::cout << "[TEST 05] Excepciones tipadas Tensor2D -> PASSED\n";
    }

    // [TEST 06] Autograd aritmetica basica (add, sub, mul, sum, mean)
    {
        auto a = TensorNode::create(Tensor2D(1, 2, {3.0, 4.0}), true);
        auto b = TensorNode::create(Tensor2D(1, 2, {2.0, 5.0}), true);
        auto out = sum(sub(mul(a, b), add(a, b)));
        out->backward();
        assert(approxEqual(out->data(0, 0), 12.0));
        assert(approxEqual(a->grad(0, 0), 1.0) && approxEqual(a->grad(0, 1), 4.0));
        assert(approxEqual(b->grad(0, 0), 2.0) && approxEqual(b->grad(0, 1), 3.0));
        std::cout << "[TEST 06] Autograd aritmetica basica -> PASSED\n";
    }

    // [TEST 07] Autograd transpose
    {
        auto x = TensorNode::create(Tensor2D(2, 3, {1, 2, 3, 4, 5, 6}), true);
        auto y = mean(transpose(x));
        y->backward();
        assert(approxEqual(x->grad(1, 2), 1.0 / 6.0));
        std::cout << "[TEST 07] Autograd transpose -> PASSED\n";
    }

    // [TEST 08] Autograd acumulacion en grafo compartido (DAG diamante)
    {
        auto x = TensorNode::create(Tensor2D(1, 1, {3.0}), true);
        auto z = add(mul(x, x), mul(x, x)); // 2 * x^2 -> dz/dx = 4x = 12
        z->backward();
        assert(approxEqual(z->data(0, 0), 18.0));
        assert(approxEqual(x->grad(0, 0), 12.0));
        std::cout << "[TEST 08] Autograd acumulacion DAG -> PASSED\n";
    }

    // [TEST 09] Autograd zeroGrad
    {
        auto x = TensorNode::create(Tensor2D(2, 2, {1, 2, 3, 4}), true);
        sum(x)->backward();
        assert(approxEqual(x->grad(0, 0), 1.0));
        x->zeroGrad();
        assert(approxEqual(x->grad(0, 0), 0.0) && approxEqual(x->grad(1, 1), 0.0));
        std::cout << "[TEST 09] Autograd zeroGrad -> PASSED\n";
    }

    // [TEST 10] Gradient Checking: matmul
    {
        auto x = TensorNode::create(Tensor2D(2, 3, {0.5, -1.2, 0.8, 1.1, -0.4, 0.3}), true);
        auto w = TensorNode::create(Tensor2D(3, 2, {0.7, -0.5, 0.2, 0.9, -0.6, 0.4}), true);
        auto loss = sum(matmul(x, w));
        loss->backward();
        auto fn = [&]() { return sum(matmul(x, w))->data(0, 0); };
        checkGrad(fn, x);
        checkGrad(fn, w);
        std::cout << "[TEST 10] Gradient Checking matmul -> PASSED\n";
    }

    // [TEST 11] Gradient Checking: rowBroadcastAdd
    {
        auto x = TensorNode::create(Tensor2D(3, 2, {0.2, -0.7, 1.3, 0.5, -0.9, 0.4}), true);
        auto b = TensorNode::create(Tensor2D(1, 2, {0.3, -0.2}), true);
        auto loss = sum(mul(rowBroadcastAdd(x, b), rowBroadcastAdd(x, b)));
        loss->backward();
        auto fn = [&]() { auto r = rowBroadcastAdd(x, b); return sum(mul(r, r))->data(0, 0); };
        checkGrad(fn, x);
        checkGrad(fn, b);
        std::cout << "[TEST 11] Gradient Checking rowBroadcastAdd -> PASSED\n";
    }

    // [TEST 12] Gradient Checking: relu
    {
        auto x = TensorNode::create(Tensor2D(2, 3, {-1.2, -0.4, 0.3, 0.8, -0.9, 1.5}), true);
        auto loss = sum(relu(x));
        loss->backward();
        checkGrad([&]() { return sum(relu(x))->data(0, 0); }, x);
        std::cout << "[TEST 12] Gradient Checking relu -> PASSED\n";
    }

    // [TEST 13] Gradient Checking: sigmoid
    {
        auto x = TensorNode::create(Tensor2D(2, 3, {-1.5, -0.3, 0.0, 0.4, 1.2, 2.0}), true);
        auto loss = sum(sigmoid(x));
        loss->backward();
        checkGrad([&]() { return sum(sigmoid(x))->data(0, 0); }, x);
        std::cout << "[TEST 13] Gradient Checking sigmoid -> PASSED\n";
    }

    // [TEST 14] Gradient Checking: tanh_op
    {
        auto x = TensorNode::create(Tensor2D(2, 3, {-1.1, -0.2, 0.0, 0.5, 1.3, -0.7}), true);
        auto loss = sum(tanh_op(x));
        loss->backward();
        checkGrad([&]() { return sum(tanh_op(x))->data(0, 0); }, x);
        std::cout << "[TEST 14] Gradient Checking tanh_op -> PASSED\n";
    }

    // [TEST 15] Gradient Checking: mseLoss
    {
        auto pred = TensorNode::create(Tensor2D(2, 2, {0.8, -0.4, 1.2, 0.1}), true);
        auto targ = TensorNode::create(Tensor2D(2, 2, {1.0, 0.0, 0.5, 0.5}), false);
        auto loss = mseLoss(pred, targ);
        loss->backward();
        checkGrad([&]() { return mseLoss(pred, targ)->data(0, 0); }, pred);
        std::cout << "[TEST 15] Gradient Checking mseLoss -> PASSED\n";
    }

    // [TEST 16] Gradient Checking y estabilidad: softmaxCrossEntropyLoss
    {
        auto logits = TensorNode::create(Tensor2D(2, 3, {1001.2, 999.5, 1000.1, 1000.3, 1002.0, 998.9}), true);
        auto targets = TensorNode::create(Tensor2D(2, 3, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}), false);
        auto loss = softmaxCrossEntropyLoss(logits, targets);
        assert(std::isfinite(loss->data(0, 0)));
        loss->backward();
        checkGrad([&]() { return softmaxCrossEntropyLoss(logits, targets)->data(0, 0); }, logits);
        std::cout << "[TEST 16] Gradient Checking softmaxCrossEntropyLoss -> PASSED\n";
    }

    // [TEST 17] Capa Linear forward y validacion de dimensiones
    {
        std::mt19937 rng(42);
        Linear lin(3, 2, rng);
        auto in = TensorNode::create(Tensor2D(4, 3, 1.0), false);
        auto out = lin.forward(in);
        assert(out->data.rows() == 4 && out->data.cols() == 2);
        assert(lin.getParameters().size() == 2);
        bool bad = false;
        try { lin.setWeightsAndBiases(Tensor2D(2, 2, 0.0), Tensor2D(1, 2, 0.0)); }
        catch (const minigrad::ShapeMismatchException&) { bad = true; }
        assert(bad);
        std::cout << "[TEST 17] Capa Linear forward y validacion -> PASSED\n";
    }

    // [TEST 18] Contenedor Sequential
    {
        std::mt19937 rng(42);
        Sequential seq;
        seq.add(std::make_shared<Linear>(2, 4, rng));
        seq.add(std::make_shared<Activation>(ActivationType::ReLU));
        seq.add(std::make_shared<Linear>(4, 1, rng));
        assert(seq.getLayers().size() == 3 && seq.getParameters().size() == 4);
        auto out = seq.forward(TensorNode::create(Tensor2D(3, 2, 0.5), false));
        assert(out->data.rows() == 3 && out->data.cols() == 1);
        std::cout << "[TEST 18] Contenedor Sequential -> PASSED\n";
    }

    // [TEST 19] Optimizador SGDOptimizer con momento
    {
        auto w = TensorNode::create(Tensor2D(1, 1, {5.0}), true);
        SGDOptimizer sgd({w}, 0.1, 0.5);
        for (int i = 0; i < 40; ++i) {
            sgd.zeroGrad();
            auto loss = mul(w, w);
            loss->backward();
            sgd.step();
        }
        assert(std::abs(w->data(0, 0)) < 0.05);
        std::cout << "[TEST 19] SGDOptimizer con momento -> PASSED\n";
    }

    // [TEST 20] Optimizador AdamOptimizer
    {
        auto w = TensorNode::create(Tensor2D(1, 2, {-3.0, 4.0}), true);
        AdamOptimizer adam({w}, 0.1);
        for (int i = 0; i < 120; ++i) {
            adam.zeroGrad();
            auto loss = sum(mul(w, w));
            loss->backward();
            adam.step();
        }
        assert(std::abs(w->data(0, 0)) < 0.05 && std::abs(w->data(0, 1)) < 0.05);
        std::cout << "[TEST 20] AdamOptimizer convergencia -> PASSED\n";
    }

    // [TEST 21] Entrenamiento real puerta logica XOR
    std::mt19937 rngXor(42);
    Sequential xorNet;
    xorNet.add(std::make_shared<Linear>(2, 8, rngXor));
    xorNet.add(std::make_shared<Activation>(ActivationType::Tanh));
    xorNet.add(std::make_shared<Linear>(8, 1, rngXor));
    auto xXor = TensorNode::create(Tensor2D(4, 2, {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}), false);
    auto yXor = TensorNode::create(Tensor2D(4, 1, {0.0, 1.0, 1.0, 0.0}), false);
    {
        AdamOptimizer adamXor(xorNet.getParameters(), 0.05);
        double finalLoss = 1.0;
        for (int epoch = 0; epoch < 600; ++epoch) {
            adamXor.zeroGrad();
            auto pred = xorNet.forward(xXor);
            auto loss = mseLoss(pred, yXor);
            finalLoss = loss->data(0, 0);
            loss->backward();
            adamXor.step();
        }
        assert(finalLoss < 0.01);
        auto predFinal = xorNet.forward(xXor);
        assert(predFinal->data(0, 0) < 0.1);
        assert(predFinal->data(1, 0) > 0.9);
        assert(predFinal->data(2, 0) > 0.9);
        assert(predFinal->data(3, 0) < 0.1);
        std::cout << "[TEST 21] Entrenamiento XOR (loss=" << finalLoss << ") -> PASSED\n";
    }

    // [TEST 22] Clasificacion multiclase sintetica (3 clusteres)
    {
        const size_t perClass = 10;
        const size_t total = perClass * 3;
        Tensor2D xData(total, 2, 0.0);
        Tensor2D yData(total, 3, 0.0);
        const double cx[3] = {-2.0, 2.0, 0.0};
        const double cy[3] = {-2.0, -2.0, 2.5};
        for (size_t c = 0; c < 3; ++c) {
            for (size_t i = 0; i < perClass; ++i) {
                size_t row = c * perClass + i;
                double ang = 0.6 * static_cast<double>(i);
                xData(row, 0) = cx[c] + 0.15 * std::cos(ang);
                xData(row, 1) = cy[c] + 0.15 * std::sin(ang);
                yData(row, c) = 1.0;
            }
        }
        auto X = TensorNode::create(xData, false);
        auto Y = TensorNode::create(yData, false);
        std::mt19937 rngClf(42);
        Sequential clf;
        clf.add(std::make_shared<Linear>(2, 12, rngClf));
        clf.add(std::make_shared<Activation>(ActivationType::ReLU));
        clf.add(std::make_shared<Linear>(12, 3, rngClf));
        AdamOptimizer adamClf(clf.getParameters(), 0.05);
        for (int epoch = 0; epoch < 250; ++epoch) {
            adamClf.zeroGrad();
            auto loss = softmaxCrossEntropyLoss(clf.forward(X), Y);
            loss->backward();
            adamClf.step();
        }
        auto logits = clf.forward(X);
        size_t correct = 0;
        for (size_t r = 0; r < total; ++r) {
            size_t bestC = 0;
            for (size_t c = 1; c < 3; ++c) {
                if (logits->data(r, c) > logits->data(r, bestC)) bestC = c;
            }
            if (yData(r, bestC) > 0.5) ++correct;
        }
        double acc = static_cast<double>(correct) / static_cast<double>(total);
        assert(acc >= 0.95);
        std::cout << "[TEST 22] Clasificacion multiclase (acc=" << (acc * 100.0) << "%) -> PASSED\n";
    }

    // [TEST 23] Serializacion ida y vuelta (saveModel / loadModel)
    {
        const std::string path = "tmp_xor_model.mgnet";
        minigrad::Serializer::saveModel(xorNet, path);
        std::mt19937 rngOther(999);
        Sequential loaded;
        loaded.add(std::make_shared<Linear>(2, 8, rngOther));
        loaded.add(std::make_shared<Activation>(ActivationType::Tanh));
        loaded.add(std::make_shared<Linear>(8, 1, rngOther));
        minigrad::Serializer::loadModel(loaded, path);
        auto pOrig = xorNet.forward(xXor);
        auto pLoad = loaded.forward(xXor);
        for (size_t r = 0; r < 4; ++r) {
            assert(approxEqual(pOrig->data(r, 0), pLoad->data(r, 0), 1e-9));
        }
        std::remove(path.c_str());
        std::cout << "[TEST 23] Serializacion ida y vuelta (< 1e-9) -> PASSED\n";
    }

    // [TEST 24] Serializacion validacion y excepciones
    {
        bool missingErr = false, corruptErr = false, mismatchErr = false;
        std::mt19937 rng(7);
        Sequential dummy;
        dummy.add(std::make_shared<Linear>(2, 8, rng));
        try { minigrad::Serializer::loadModel(dummy, "no_existe_12345.mgnet"); }
        catch (const minigrad::SerializationException&) { missingErr = true; }

        const std::string badPath = "tmp_corrupt.mgnet";
        { std::ofstream f(badPath); f << "CABECERA_FALSA\n"; }
        try { minigrad::Serializer::loadModel(dummy, badPath); }
        catch (const minigrad::SerializationException&) { corruptErr = true; }
        std::remove(badPath.c_str());

        const std::string validPath = "tmp_valid.mgnet";
        minigrad::Serializer::saveModel(xorNet, validPath);
        Sequential wrongShape;
        wrongShape.add(std::make_shared<Linear>(2, 4, rng));
        wrongShape.add(std::make_shared<Activation>(ActivationType::Tanh));
        wrongShape.add(std::make_shared<Linear>(4, 1, rng));
        try { minigrad::Serializer::loadModel(wrongShape, validPath); }
        catch (const minigrad::SerializationException&) { mismatchErr = true; }
        std::remove(validPath.c_str());

        assert(missingErr && corruptErr && mismatchErr);
        std::cout << "[TEST 24] Serializacion excepciones y validacion -> PASSED\n";
    }

    std::cout << "\n24/24 tests ejecutados exitosamente.\n";
    return 0;
}
