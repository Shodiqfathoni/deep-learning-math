# Deep Learning from Scratch: Sigmoid Network with 2 Hidden Layers

A step-by-step, fully expanded (no simplifications) derivation and
computation of forward propagation and backpropagation for a feedforward
neural network using **sigmoid activation** and **Mean Squared Error
(MSE)** loss.

------------------------------------------------------------------------

## 1. Problem Setup

**Dataset (single sample for clarity)**

  x1   x2   y
  ---- ---- ---
  1    1    1

**Objective**: Learn parameters to predict ( y `\in `{=tex}{0,1} ) from
inputs (x_1, x_2).

------------------------------------------------------------------------

## 2. Model Architecture

-   Input layer: 2 features
-   Hidden Layer 1: 2 neurons (Sigmoid)
-   Hidden Layer 2: 2 neurons (Sigmoid)
-   Output Layer: 1 neuron (Sigmoid)

------------------------------------------------------------------------

## 3. Notation

-   Linear: ( z = W a + b )
-   Activation (sigmoid): ( a = `\sigma`{=tex}(z) =
    `\frac{1}{1 + e^{-z}}`{=tex} )
-   Sigmoid derivative: ( `\sigma`{=tex}'(z) = a(1-a) )
-   Loss (MSE): ( L = `\frac{1}{2}`{=tex}(`\hat{y}`{=tex} - y)\^2 )

------------------------------------------------------------------------

## 4. Parameter Initialization

### Hidden Layer 1

-   (w_1=0.5,; w_2=0.4)
-   (w_3=0.3,; w_4=0.2)
-   (b_1=0,; b_2=0)

### Hidden Layer 2

-   (w_5=0.6,; w_6=0.7)
-   (w_7=0.8,; w_8=0.9)
-   (b_3=0,; b_4=0)

### Output Layer

-   (w_9=0.5,; w\_{10}=0.6)
-   (b_5=0)

Learning rate: ( `\alpha `{=tex}= 0.1 )

------------------------------------------------------------------------

## 5. Forward Propagation

### Hidden Layer 1

\[ z\_{1}\^{(1)} = 0.5(1) + 0.4(1) = 0.9, `\quad `{=tex}a\_{1}\^{(1)} =
`\sigma`{=tex}(0.9) `\approx 0.71`{=tex} \] \[ z\_{2}\^{(1)} = 0.3(1) +
0.2(1) = 0.5, `\quad `{=tex}a\_{2}\^{(1)} = `\sigma`{=tex}(0.5)
`\approx 0.62`{=tex} \]

### Hidden Layer 2

\[ z\_{1}\^{(2)} = 0.6(0.71) + 0.7(0.62) = 0.86,
`\quad `{=tex}a\_{1}\^{(2)} = `\sigma`{=tex}(0.86) `\approx 0.70`{=tex}
\] \[ z\_{2}\^{(2)} = 0.8(0.71) + 0.9(0.62) = 1.09,
`\quad `{=tex}a\_{2}\^{(2)} = `\sigma`{=tex}(1.09) `\approx 0.75`{=tex}
\]

### Output Layer

\[ z\^{(3)} = 0.5(0.70) + 0.6(0.75) = 0.80,
`\quad `{=tex}`\hat{y}`{=tex} = a\^{(3)} = `\sigma`{=tex}(0.80)
`\approx 0.69`{=tex} \]

------------------------------------------------------------------------

## 6. Loss

\[ L = `\frac{1}{2}`{=tex}(`\hat{y}`{=tex} - y)\^2 =
`\frac{1}{2}`{=tex}(0.69 - 1)\^2 `\approx 0.048`{=tex} \]

------------------------------------------------------------------------

## 7. Backpropagation (Full Chain Rule)

### Output Layer

\[ `\frac{\partial L}{\partial a^{(3)}}`{=tex} = (`\hat{y}`{=tex} - y) =
-0.31 \]

\[ `\frac{\partial a^{(3)}}{\partial z^{(3)}}`{=tex} =
a^{(3)}(1-a^{(3)}) = 0.69(0.31) = 0.2139 \]

\[ `\delta`{=tex}\^{(3)} = `\frac{\partial L}{\partial z^{(3)}}`{=tex} =
-0.31 `\times 0.2139`{=tex} = -0.0663 \]

Gradients: \[ `\frac{\partial L}{\partial w_9}`{=tex} =
`\delta`{=tex}\^{(3)} a\_{1}\^{(2)} = -0.0663(0.70) = -0.0464 \] \[
`\frac{\partial L}{\partial w_{10}}`{=tex} = `\delta`{=tex}\^{(3)}
a\_{2}\^{(2)} = -0.0663(0.75) = -0.0497 \]

------------------------------------------------------------------------

### Hidden Layer 2

\[ `\delta`{=tex}*{1}\^{(2)} = `\delta`{=tex}\^{(3)} w_9
`\cdot `{=tex}a*{1}^{(2)}(1-a\_{1}^{(2)}) = (-0.0663)(0.5)(0.70)(0.30) =
-0.00696 \] \[ `\delta`{=tex}*{2}\^{(2)} = `\delta`{=tex}\^{(3)} w*{10}
`\cdot `{=tex}a\_{2}^{(2)}(1-a\_{2}^{(2)}) = (-0.0663)(0.6)(0.75)(0.25)
= -0.00746 \]

Gradients: \[ `\frac{\partial L}{\partial w_5}`{=tex} =
`\delta`{=tex}*{1}\^{(2)} a*{1}\^{(1)}, `\quad`{=tex}
`\frac{\partial L}{\partial w_6}`{=tex} = `\delta`{=tex}*{1}\^{(2)}
a*{2}\^{(1)} \] \[ `\frac{\partial L}{\partial w_7}`{=tex} =
`\delta`{=tex}*{2}\^{(2)} a*{1}\^{(1)}, `\quad`{=tex}
`\frac{\partial L}{\partial w_8}`{=tex} = `\delta`{=tex}*{2}\^{(2)}
a*{2}\^{(1)} \]

------------------------------------------------------------------------

### Hidden Layer 1

\[ `\delta`{=tex}*{1}\^{(1)} = (`\delta`{=tex}*{1}\^{(2)} w_5 +
`\delta`{=tex}*{2}\^{(2)} w_7) `\cdot `{=tex}a*{1}^{(1)}(1-a\_{1}^{(1)})
\] \[ `\delta`{=tex}*{2}\^{(1)} = (`\delta`{=tex}*{1}\^{(2)} w_6 +
`\delta`{=tex}*{2}\^{(2)} w_8) `\cdot `{=tex}a*{2}^{(1)}(1-a\_{2}^{(1)})
\]

Gradients: \[ `\frac{\partial L}{\partial w_1}`{=tex} =
`\delta`{=tex}*{1}\^{(1)} x_1, `\quad`{=tex}
`\frac{\partial L}{\partial w_2}`{=tex} = `\delta`{=tex}*{1}\^{(1)} x_2
\] \[ `\frac{\partial L}{\partial w_3}`{=tex} =
`\delta`{=tex}*{2}\^{(1)} x_1, `\quad`{=tex}
`\frac{\partial L}{\partial w_4}`{=tex} = `\delta`{=tex}*{2}\^{(1)} x_2
\]

------------------------------------------------------------------------

## 8. Parameter Update

\[ w `\leftarrow `{=tex}w -
`\alpha `{=tex}`\frac{\partial L}{\partial w}`{=tex} \]

------------------------------------------------------------------------

## 9. Key Takeaways

-   Sigmoid derivative (a(1-a)) appears **explicitly in every layer
    during backpropagation**
-   Deep learning relies on **chain rule composition across layers**
-   No simplification (unlike logistic regression with BCE) --- full
    gradients are preserved

------------------------------------------------------------------------

## 10. Reproducibility

This document is suitable as: - Educational reference - Portfolio
demonstration of understanding backpropagation - Basis for implementing
neural networks from scratch

------------------------------------------------------------------------
