# MiniGradCPP

Motor modular de redes neuronales y diferenciacion automatica (Autograd) implementado desde cero en **C++17** sin dependencias externas.

## Caracteristicas principales
- **Tensor2D**: operaciones matriciales (`matmul`, `transpose`), *broadcasting* de sesgos e inicializacion determinista Xavier/He.
- **Autograd**: grafo computacional dinamico con ordenacion topologica y retropropagacion (`backward`) sobre `relu`, `sigmoid`, `tanh_op`, `mseLoss` y `softmaxCrossEntropyLoss`.
- **Redes y optimizadores**: capas densas (`Linear`), activaciones (`Activation`), contenedor `Sequential`, `SGDOptimizer` con momento y `AdamOptimizer`.
- **Serializacion nativa**: guardado y carga de pesos en formato propio con validacion estricta (`Serializer.hpp` / `Serializer.cpp`).
- **Suite de pruebas**: 24 tests unitarios con `assert` incluyendo *gradient checking* numerico, aprendizaje de XOR y clasificacion multiclase.

## Compilacion y ejecucion

```bash
g++ -std=c++17 -Wall -Wextra -pedantic main.cpp Serializer.cpp -o MiniGrad.exe
./MiniGrad.exe
```
