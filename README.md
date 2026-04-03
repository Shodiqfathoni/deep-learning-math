# Deep Learning from Scratch: Sigmoid Network (2 Hidden Layers)

A clean, GitHub-friendly walkthrough of forward propagation and
backpropagation **without simplification** using sigmoid activation and
MSE loss.

------------------------------------------------------------------------

## 1. Dataset

  x1   x2   y
  ---- ---- ---
  1    1    1

------------------------------------------------------------------------

## 2. Architecture

-   Input: 2 features\
-   Hidden Layer 1: 2 neurons (sigmoid)\
-   Hidden Layer 2: 2 neurons (sigmoid)\
-   Output: 1 neuron (sigmoid)

------------------------------------------------------------------------

## 3. Activation Function

sigmoid(z) = 1 / (1 + e\^(-z))\
sigmoid'(z) = a(z) \* (1 - a(z))

------------------------------------------------------------------------

## 4. Initialization

Hidden Layer 1\
w1=0.5, w2=0.4\
w3=0.3, w4=0.2\
b1=0, b2=0

Hidden Layer 2\
w5=0.6, w6=0.7\
w7=0.8, w8=0.9\
b3=0, b4=0

Output Layer\
w9=0.5, w10=0.6\
b5=0

Learning rate = 0.1

------------------------------------------------------------------------

## 5. Forward Pass

### Hidden Layer 1

z1(1) = 0.5(1) + 0.4(1) = 0.9\
a1(1) = sigmoid(0.9) ≈ 0.71

z2(1) = 0.3(1) + 0.2(1) = 0.5\
a2(1) = sigmoid(0.5) ≈ 0.62

------------------------------------------------------------------------

### Hidden Layer 2

z1(2) = 0.6(0.71) + 0.7(0.62) = 0.86\
a1(2) = sigmoid(0.86) ≈ 0.70

z2(2) = 0.8(0.71) + 0.9(0.62) = 1.09\
a2(2) = sigmoid(1.09) ≈ 0.75

------------------------------------------------------------------------

### Output Layer

z(3) = 0.5(0.70) + 0.6(0.75) = 0.80\
y_hat = sigmoid(0.80) ≈ 0.69

------------------------------------------------------------------------

## 6. Loss (MSE)

L = 1/2 (y_hat - y)\^2\
L = 1/2 (0.69 - 1)\^2 ≈ 0.048

------------------------------------------------------------------------

## 7. Backpropagation

### Output Layer

dL/da3 = (y_hat - y) = -0.31

da3/dz3 = a3(1 - a3) = 0.69(0.31) = 0.2139

delta3 = -0.31 × 0.2139 = -0.0663

dw9 = delta3 × a1(2) = -0.0464\
dw10 = delta3 × a2(2) = -0.0497

------------------------------------------------------------------------

### Hidden Layer 2

delta2_1 = delta3 × w9 × a1(2)(1 - a1(2))\
= (-0.0663)(0.5)(0.70)(0.30) = -0.00696

delta2_2 = delta3 × w10 × a2(2)(1 - a2(2))\
= (-0.0663)(0.6)(0.75)(0.25) = -0.00746

dw5 = delta2_1 × a1(1)\
dw6 = delta2_1 × a2(1)

dw7 = delta2_2 × a1(1)\
dw8 = delta2_2 × a2(1)

------------------------------------------------------------------------

### Hidden Layer 1

delta1_1 = (delta2_1*w5 + delta2_2*w7) × a1(1)(1 - a1(1))

delta1_2 = (delta2_1*w6 + delta2_2*w8) × a2(1)(1 - a2(1))

dw1 = delta1_1 × x1\
dw2 = delta1_1 × x2

dw3 = delta1_2 × x1\
dw4 = delta1_2 × x2

------------------------------------------------------------------------

## 8. Update Rule

w = w - alpha × gradient

------------------------------------------------------------------------

## 9. Key Insights

-   Sigmoid derivative appears in every backprop step\
-   Chain rule is applied layer-by-layer\
-   No simplification like logistic regression

------------------------------------------------------------------------

## 10. Author Note

This project demonstrates a deep understanding of: - Forward
propagation\
- Backpropagation\
- Gradient flow in deep neural networks

Suitable for portfolio and educational purposes.
