# Deep Learning with Sigmoid (2 Hidden Layers) - Step by Step (No Simplification)

## Dataset

  x1   x2   y
  ---- ---- ---
  1    1    1

------------------------------------------------------------------------

## Arsitektur Model

-   Input Layer: 2 neuron
-   Hidden Layer 1: 2 neuron (Sigmoid)
-   Hidden Layer 2: 2 neuron (Sigmoid)
-   Output Layer: 1 neuron (Sigmoid)

------------------------------------------------------------------------

## Inisialisasi Parameter

### Hidden Layer 1

w1 = 0.5, w2 = 0.4\
w3 = 0.3, w4 = 0.2\
b1 = 0, b2 = 0

### Hidden Layer 2

w5 = 0.6, w6 = 0.7\
w7 = 0.8, w8 = 0.9\
b3 = 0, b4 = 0

### Output Layer

w9 = 0.5, w10 = 0.6\
b5 = 0

Learning rate = 0.1

------------------------------------------------------------------------

## Forward Pass

### Hidden Layer 1

z1_1 = (0.5)(1) + (0.4)(1) = 0.9\
a1_1 = 1 / (1 + e\^(-0.9)) = 0.71

z1_2 = (0.3)(1) + (0.2)(1) = 0.5\
a1_2 = 1 / (1 + e\^(-0.5)) = 0.62

------------------------------------------------------------------------

### Hidden Layer 2

z2_1 = (0.6)(0.71) + (0.7)(0.62) = 0.86\
a2_1 = 1 / (1 + e\^(-0.86)) = 0.70

z2_2 = (0.8)(0.71) + (0.9)(0.62) = 1.09\
a2_2 = 1 / (1 + e\^(-1.09)) = 0.75

------------------------------------------------------------------------

### Output Layer

z3 = (0.5)(0.70) + (0.6)(0.75) = 0.80\
a3 = 1 / (1 + e\^(-0.80)) = 0.69

Prediction = 0.69

------------------------------------------------------------------------

## Loss (MSE)

L = 1/2 (a3 - y)\^2\
L = 1/2 (0.69 - 1)\^2 = 0.048

------------------------------------------------------------------------

## Backpropagation (FULL DETAIL)

### Step 1: dL/da3

dL/da3 = (a3 - y) = -0.31

------------------------------------------------------------------------

### Step 2: Sigmoid Derivative Output

da3/dz3 = a3(1 - a3)\
= 0.69(1 - 0.69) = 0.2139

------------------------------------------------------------------------

### Step 3: delta output

delta3 = -0.31 × 0.2139 = -0.0663

------------------------------------------------------------------------

### Step 4: Gradient Output Weights

dw9 = delta3 × a2_1 = -0.0464\
dw10 = delta3 × a2_2 = -0.0497

------------------------------------------------------------------------

### Step 5: Backprop to Hidden Layer 2

delta2_1 = delta3 × w9 × a2_1(1 - a2_1)\
= (-0.0663)(0.5)(0.70)(0.30) = -0.00696

delta2_2 = delta3 × w10 × a2_2(1 - a2_2)\
= (-0.0663)(0.6)(0.75)(0.25) = -0.00746

------------------------------------------------------------------------

### Step 6: Gradient Hidden Layer 2

dw5 = delta2_1 × a1_1\
dw6 = delta2_1 × a1_2

dw7 = delta2_2 × a1_1\
dw8 = delta2_2 × a1_2

------------------------------------------------------------------------

### Step 7: Backprop to Hidden Layer 1

delta1_1 = (delta2_1*w5 + delta2_2*w7) × a1_1(1 - a1_1)

delta1_2 = (delta2_1*w6 + delta2_2*w8) × a1_2(1 - a1_2)

------------------------------------------------------------------------

### Step 8: Gradient Hidden Layer 1

dw1 = delta1_1 × x1\
dw2 = delta1_1 × x2

dw3 = delta1_2 × x1\
dw4 = delta1_2 × x2

------------------------------------------------------------------------

## Update Weight

w = w - α × gradient

------------------------------------------------------------------------

## Insight Penting

-   Turunan sigmoid muncul di SETIAP layer saat backprop
-   Tidak ada simplifikasi seperti logistic regression
-   Chain rule digunakan penuh

------------------------------------------------------------------------

## Kesimpulan

-   Forward → hanya sigmoid\
-   Backprop → sigmoid derivative muncul\
-   Deep learning = chain rule berlapis
