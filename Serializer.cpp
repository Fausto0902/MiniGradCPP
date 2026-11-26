#include "Serializer.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <iomanip>
#include <type_traits>
#include <utility>
namespace minigrad {
namespace {
template <typename T, typename = void> struct HasConstCall : std::false_type {};
template <typename T> struct HasConstCall<T, std::void_t<decltype(std::declval<const T&>()(std::size_t{0}, std::size_t{0}))>> : std::true_type {};
template <typename T, typename = void> struct HasMutCall : std::false_type {};
template <typename T> struct HasMutCall<T, std::void_t<decltype(std::declval<T&>()(std::size_t{0}, std::size_t{0}) = 0.0)>> : std::true_type {};
double getElem(const Tensor2D& t, std::size_t r, std::size_t c) { if constexpr (HasConstCall<Tensor2D>::value) return t(r, c); else return t.at(r, c); }
void setElem(Tensor2D& t, std::size_t r, std::size_t c, double v) { if constexpr (HasMutCall<Tensor2D>::value) t(r, c) = v; else t.at(r, c) = v; }
std::vector<std::shared_ptr<Linear>> getLinearLayers(const Sequential& model) { std::vector<std::shared_ptr<Linear>> out; for (const auto& l : model.getLayers()) { if (auto lin = std::dynamic_pointer_cast<Linear>(l)) out.push_back(lin); } return out; }
}
void Serializer::saveModel(const Sequential& model, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) throw minigrad::SerializationException("No se pudo abrir archivo: " + filepath);
    auto layers = getLinearLayers(model);
    file << "MINIGRAD_MODEL_V1\nNUM_LINEAR_LAYERS " << layers.size() << "\n" << std::setprecision(17);
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const Tensor2D& w = layers[i]->getWeight()->data;
        const Tensor2D& b = layers[i]->getBias()->data;
        file << "LAYER " << i << " " << w.rows() << " " << w.cols() << " " << b.rows() << " " << b.cols() << "\nWEIGHTS\n";
        for (std::size_t r = 0; r < w.rows(); ++r) { for (std::size_t c = 0; c < w.cols(); ++c) file << getElem(w, r, c) << (c + 1 < w.cols() ? " " : ""); file << "\n"; }
        file << "BIASES\n";
        for (std::size_t r = 0; r < b.rows(); ++r) { for (std::size_t c = 0; c < b.cols(); ++c) file << getElem(b, r, c) << (c + 1 < b.cols() ? " " : ""); file << "\n"; }
    }
    if (!file.good()) throw minigrad::SerializationException("Error de escritura en: " + filepath);
}
void Serializer::loadModel(Sequential& model, const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw minigrad::SerializationException("No se pudo abrir archivo: " + filepath);
    std::string hdr, tag;
    std::size_t count = 0;
    if (!(file >> hdr) || hdr != "MINIGRAD_MODEL_V1") throw minigrad::SerializationException("Cabecera invalida");
    if (!(file >> tag >> count) || tag != "NUM_LINEAR_LAYERS") throw minigrad::SerializationException("Conteo invalido");
    auto layers = getLinearLayers(model);
    if (layers.size() != count) throw minigrad::SerializationException("Numero de capas incompatible");
    std::vector<std::pair<Tensor2D, Tensor2D>> parsed;
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t idx = 0, wr = 0, wc = 0, br = 0, bc = 0;
        if (!(file >> tag >> idx >> wr >> wc >> br >> bc) || tag != "LAYER" || idx != i) throw minigrad::SerializationException("Capa invalida");
        const Tensor2D& cw = layers[i]->getWeight()->data;
        const Tensor2D& cb = layers[i]->getBias()->data;
        if (wr != cw.rows() || wc != cw.cols() || br != cb.rows() || bc != cb.cols()) throw minigrad::SerializationException("Dimensiones incompatibles");
        if (!(file >> tag) || tag != "WEIGHTS") throw minigrad::SerializationException("Falta bloque WEIGHTS");
        Tensor2D nw(wr, wc, 0.0);
        for (std::size_t r = 0; r < wr; ++r) for (std::size_t c = 0; c < wc; ++c) { double v = 0.0; if (!(file >> v) || !std::isfinite(v)) throw minigrad::SerializationException("Peso corrupto"); setElem(nw, r, c, v); }
        if (!(file >> tag) || tag != "BIASES") throw minigrad::SerializationException("Falta bloque BIASES");
        Tensor2D nb(br, bc, 0.0);
        for (std::size_t r = 0; r < br; ++r) for (std::size_t c = 0; c < bc; ++c) { double v = 0.0; if (!(file >> v) || !std::isfinite(v)) throw minigrad::SerializationException("Sesgo corrupto"); setElem(nb, r, c, v); }
        parsed.emplace_back(std::move(nw), std::move(nb));
    }
    std::string extra;
    if (file >> extra) throw minigrad::SerializationException("Datos extra inesperados");
    for (std::size_t i = 0; i < count; ++i) layers[i]->setWeightsAndBiases(parsed[i].first, parsed[i].second);
}
void Serializer::save(const std::shared_ptr<Sequential>& m, const std::string& f) { if (!m) throw minigrad::SerializationException("Modelo nulo"); saveModel(*m, f); }
void Serializer::load(const std::shared_ptr<Sequential>& m, const std::string& f) { if (!m) throw minigrad::SerializationException("Modelo nulo"); loadModel(*m, f); }
}
