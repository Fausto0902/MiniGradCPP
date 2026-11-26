#ifndef MINIGRAD_EXCEPTIONS_HPP
#define MINIGRAD_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>

/**
 * @namespace minigrad
 * @brief Espacio de nombres principal para el motor de diferenciación automática y redes neuronales.
 */
namespace minigrad {

/**
 * @class MiniGradException
 * @brief Clase base para todas las excepciones personalizadas dentro del motor MiniGrad.
 * Hereda de std::runtime_error para integrarse con el manejo de excepciones estándar de C++.
 */
class MiniGradException : public std::runtime_error {
public:
    /**
     * @brief Constructor de la excepción base.
     * @param message Mensaje descriptivo del error.
     */
    explicit MiniGradException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @class ShapeMismatchException
 * @brief Excepción lanzada cuando las dimensiones de los tensores no son compatibles para una operación.
 * Ejemplo: Multiplicación de matrices donde las columnas de A no coinciden con las filas de B,
 * o sumas de tensores con formas incompatibles sin reglas de broadcasting válidas.
 */
class ShapeMismatchException : public MiniGradException {
public:
    /**
     * @brief Constructor para errores de dimensiones.
     * @param message Detalle técnico de la discrepancia de formas.
     */
    explicit ShapeMismatchException(const std::string& message)
        : MiniGradException("[ShapeMismatchException] " + message) {}

    /**
     * @brief Constructor especializado para facilitar el reporte de dimensiones.
     * @param expected Descripción de la forma esperada.
     * @param actual Descripción de la forma recibida.
     */
    ShapeMismatchException(const std::string& expected, const std::string& actual)
        : MiniGradException("[ShapeMismatchException] Expected shape: " + expected + ", but got: " + actual) {}
};

/**
 * @class InvalidArgumentException
 * @brief Excepción lanzada cuando se pasan argumentos fuera de rango o inválidos a funciones o métodos.
 * Ejemplo: Índices de acceso fuera de límites, tasas de aprendizaje negativas, o parámetros de optimización inválidos.
 */
class InvalidArgumentException : public MiniGradException {
public:
    /**
     * @brief Constructor para argumentos inválidos.
     * @param message Detalle del argumento que causó el fallo.
     */
    explicit InvalidArgumentException(const std::string& message)
        : MiniGradException("[InvalidArgumentException] " + message) {}
};

/**
 * @class SerializationException
 * @brief Excepción lanzada durante los procesos de guardado o carga de modelos en disco.
 * Se activa ante archivos corruptos, versiones incompatibles, errores de permisos de E/S
 * o discrepancias entre la arquitectura del modelo en memoria y los datos serializados.
 */
class SerializationException : public MiniGradException {
public:
    /**
     * @brief Constructor para errores de serialización/deserialización.
     * @param message Detalle del error de persistencia.
     */
    explicit SerializationException(const std::string& message)
        : MiniGradException("[SerializationException] " + message) {}
};

} // namespace minigrad

#endif // MINIGRAD_EXCEPTIONS_HPP
