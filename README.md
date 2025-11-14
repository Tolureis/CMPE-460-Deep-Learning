# CMPE-460-Deep-Learning
CMPE 460 Deep Learning University Lesson

These Notebooks has taken from https://github.com/udlbook/udlbook/tree/main/Notebooks and completed for this lesson.

FORMULAS FOR GRADIENT DESCENT

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Formula / Operation</th>
<th>LaTeX Representation</th>
<th>Code Calculation Line</th>
</tr>
</thead>
<tbody>

<tr>
<td><b>Linear Model</b></td>
<td>
\[
\hat{y} = \phi_0 + \phi_1 x
\]
</td>
<td>
<pre>pred_y = phi[0] + phi[1] * data_x</pre>
</td>
</tr>

<tr>
<td><b>Loss Function (SSE)</b></td>
<td>
\[
L = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]
</td>
<td>
<pre>loss = np.sum((data_y - pred_y)**2)</pre>
</td>
</tr>

<tr>
<td><b>Error (Residual)</b></td>
<td>
\[
e_i = \hat{y}_i - y_i
\]
</td>
<td>
<pre>error = y_pred - data_y</pre>
</td>
</tr>

<tr>
<td><b>Derivative (for φ₀)</b></td>
<td>
\[
\frac{\partial L}{\partial \phi_0}
= 2\sum e_i
\]
</td>
<td>
<pre>dl_dphi0 = 2 * np.sum(error)</pre>
</td>
</tr>

<tr>
<td><b>Derivative (for φ₁)</b></td>
<td>
\[
\frac{\partial L}{\partial \phi_1}
= 2\sum e_i x_i
\]
</td>
<td>
<pre>dl_dphi1 = 2 * np.sum(error * data_x)</pre>
</td>
</tr>

<tr>
<td><b>Gradient Vector</b></td>
<td>
\[
\nabla L =
\begin{bmatrix}
\partial L/\partial \phi_0 \\
\partial L/\partial \phi_1
\end{bmatrix}
\]
</td>
<td>
<pre>gradient = np.array([[dl_dphi0],[dl_dphi1]])</pre>
</td>
</tr>

<tr>
<td><b>Normalized Negative Gradient</b></td>
<td>
\[
d = -\frac{\nabla L}{\|\nabla L\|}
\]
</td>
<td>
<pre>direction = -gradient / np.linalg.norm(gradient)</pre>
</td>
</tr>

<tr>
<td><b>Update Formula</b></td>
<td>
\[
\phi_{\text{new}} = \phi + \alpha d
\]
</td>
<td>
<pre>phi_new = phi + alpha * direction</pre>
</td>
</tr>

<tr>
<td><b>1D Line Search</b></td>
<td>
\[
L(\alpha)=L(\phi+\alpha d)
\]
</td>
<td>
<pre>phi_start + search_direction * dist_prop</pre>
</td>
</tr>

<tr>
<td><b>Loss Surface Value</b></td>
<td>
\[
L(\phi_0,\phi_1)
\]
</td>
<td>
<pre>loss_mesh[idx] = compute_loss(...)</pre>
</td>
</tr>

</tbody>
</table>

FORMULAS FOR STOCHASTIC GRADIENT DESCENT

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Operation</th>
<th>Mathematical Formula (Clean LaTeX)</th>
<th>Code Line</th>
</tr>
</thead>
<tbody>

<tr>
<td><b>Transformed Input</b></td>
<td>
\[
z = \phi_0 + 0.06\,\phi_1\,x
\]
</td>
<td>
<pre>z = phi[0] + 0.06 * phi[1] * x</pre>
</td>
</tr>

<tr>
<td><b>Sine Component</b></td>
<td>
\[
\sin(z)
\]
</td>
<td>
<pre>sin_component = np.sin(z)</pre>
</td>
</tr>

<tr>
<td><b>Gaussian Component</b></td>
<td>
\[
\exp\!\left(-\frac{z^{2}}{32}\right)
\]
</td>
<td>
<pre>gauss_component = np.exp(-(z*z)/32)</pre>
</td>
</tr>

<tr>
<td><b>Model Output</b></td>
<td>
\[
\hat{y} = \sin(z)\,\exp\!\left(-\frac{z^{2}}{32}\right)
\]
</td>
<td>
<pre>y_pred = sin_component * gauss_component</pre>
</td>
</tr>

<tr>
<td><b>Loss Function (SSE)</b></td>
<td>
\[
L = \sum_{i=1}^N (\hat{y}_{i} - y_{i})^{2}
\]
</td>
<td>
<pre>loss = np.sum((y_pred - data_y)**2)</pre>
</td>
</tr>

<tr>
<td><b>Residual</b></td>
<td>
\[
e_i = \hat{y}_i - y_i
\]
</td>
<td>
<pre>error = y_pred - data_y</pre>
</td>
</tr>

<tr>
<td><b>Derivative of Model w.r.t φ₀</b></td>
<td>
\[
\frac{\partial \hat{y}}{\partial \phi_0}
=
\cos(z)\,\exp\!\left(-\frac{z^{2}}{32}\right)
-
\sin(z)\,\exp\!\left(-\frac{z^{2}}{32}\right)
\cdot
\frac{z}{16}
\]
</td>
<td>
<pre>cos(z)*gauss - sin(z)*gauss*(z/16)</pre>
</td>
</tr>

<tr>
<td><b>Derivative of Loss w.r.t φ₀</b></td>
<td>
\[
\frac{\partial L}{\partial \phi_0}
=
2\sum_{i}
e_i
\left(
\frac{\partial \hat{y}_i}{\partial \phi_0}
\right)
\]
</td>
<td>
<pre>2 * deriv * (y_pred - y)</pre>
</td>
</tr>

<tr>
<td><b>Derivative of Model w.r.t φ₁</b></td>
<td>
\[
\frac{\partial \hat{y}}{\partial \phi_1}
=
0.06 x\,
\left[
\cos(z)\,\exp\!\left(-\frac{z^{2}}{32}\right)
-
\sin(z)\,\exp\!\left(-\frac{z^{2}}{32}\right)
\cdot
\frac{z}{16}
\right]
\]
</td>
<td>
<pre>0.06*x*(cos(z)*gauss - sin(z)*gauss*(z/16))</pre>
</td>
</tr>

<tr>
<td><b>Derivative of Loss w.r.t φ₁</b></td>
<td>
\[
\frac{\partial L}{\partial \phi_1}
=
2\sum_i e_i
\left(
\frac{\partial \hat{y}_i}{\partial \phi_1}
\right)
\]
</td>
<td>
<pre>2*deriv*(y_pred - y)</pre>
</td>
</tr>

<tr>
<td><b>Gradient Vector</b></td>
<td>
\[
\nabla L =
\begin{bmatrix}
\frac{\partial L}{\partial \phi_0}\\[4pt]
\frac{\partial L}{\partial \phi_1}
\end{bmatrix}
\]
</td>
<td>
<pre>gradient = [[dl_dphi0],[dl_dphi1]]</pre>
</td>
</tr>

<tr>
<td><b>Gradient Descent Update</b></td>
<td>
\[
\phi \leftarrow \phi - \alpha \nabla L
\]
</td>
<td>
<pre>phi = phi - alpha * gradient</pre>
</td>
</tr>

<tr>
<td><b>Mini-Batch Sampling</b></td>
<td>
\[
B = \{(x_i, y_i)\ :\ i\in \text{random subset}\}
\]
</td>
<td>
<pre>indices = np.random.permutation(n_data)[:batch_size]</pre>
</td>
</tr>

<tr>
<td><b>SGD Update Rule</b></td>
<td>
\[
\phi \leftarrow \phi - \alpha \nabla L_B
\]
</td>
<td>
<pre>phi = phi - alpha * gradient</pre>
</td>
</tr>

</tbody>
</table>

FORMULAS FOR LINE SEARCH

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Operation</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Code Line</th>
</tr>
</thead>
<tbody>

<tr>
<td><b>Loss Function</b></td>
<td>
\[
L(\phi)
= 1
- 0.5\, e^{ -\frac{(\phi - 0.65)^2}{0.1} }
- 0.45\, e^{ -\frac{(\phi - 0.35)^2}{0.02} }
\]
</td>
<td>
<pre>return 1 - 0.5*np.exp(-(phi-0.65)**2/0.1) - 0.45*np.exp(-(phi-0.35)**2/0.02)</pre>
</td>
</tr>

<tr>
<td><b>Initial Interval Points</b></td>
<td>
\[
a=0,\quad b=0.33,\quad c=0.66,\quad d=1
\]
</td>
<td>
<pre>a=0; b=0.33; c=0.66; d=1.0</pre>
</td>
</tr>

<tr>
<td><b>Evaluate Function Values</b></td>
<td>
\[
L(a),\,L(b),\,L(c),\,L(d)
\]
</td>
<td>
<pre>lossa = loss_function(a)</pre>
<pre>lossb = loss_function(b)</pre>
<pre>lossc = loss_function(c)</pre>
<pre>lossd = loss_function(d)</pre>
</td>
</tr>

<tr>
<td><b>Rule #1 (If A is the minimum)</b></td>
<td>
\[
\text{If } L(a) < L(b),L(c),L(d):\quad
\begin{aligned}
b' &= a + \frac{b-a}{2} \\
c' &= a + \frac{c-a}{2} \\
d' &= a + \frac{d-a}{2}
\end{aligned}
\]
</td>
<td>
<pre>b = a + (b - a) / 2</pre>
<pre>c = a + (c - a) / 2</pre>
<pre>d = a + (d - a) / 2</pre>
</td>
</tr>

<tr>
<td><b>Rule #2 (If B < C)</b></td>
<td>
\[
\text{If } L(b) < L(c):\quad
\begin{aligned}
d' &= c \\
b' &= a + \frac{1}{3}(d'-a) \\
c' &= a + \frac{2}{3}(d'-a)
\end{aligned}
\]
</td>
<td>
<pre>d = c</pre>
<pre>b = a + (d - a) / 3</pre>
<pre>c = a + 2 * (d - a) / 3</pre>
</td>
</tr>

<tr>
<td><b>Rule #3 (If C < B)</b></td>
<td>
\[
\text{If } L(c) < L(b):\quad
\begin{aligned}
a' &= b \\
b' &= a' + \frac{1}{3}(d-a') \\
c' &= a' + \frac{2}{3}(d-a')
\end{aligned}
\]
</td>
<td>
<pre>a = b</pre>
<pre>b = a + (d - a) / 3</pre>
<pre>c = a + 2 * (d - a) / 3</pre>
</td>
</tr>

<tr>
<td><b>Stopping Criterion</b></td>
<td>
\[
|b-c| \le \text{threshold}
\quad\text{or}\quad
\text{iteration limit reached}
\]
</td>
<td>
<pre>while np.abs(b-c) > thresh and n_iter < max_iter:</pre>
</td>
</tr>

<tr>
<td><b>Final Solution</b></td>
<td>
\[
\phi^\* = \frac{b + c}{2}
\]
</td>
<td>
<pre>soln = (b + c) / 2</pre>
</td>
</tr>

</tbody>
</table>

FORMULAS FOR MOMENTUM

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Operation</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Code Line (Exact Calculation)</th>
</tr>
</thead>
<tbody>

<!-- MODEL -->

<tr>
<td><b>Transformed Input</b></td>
<td>
\[
z = \phi_0 + 0.06\,\phi_1\,x
\]
</td>
<td><pre>z = phi[0] + 0.06 * phi[1] * x</pre></td>
</tr>

<tr>
<td><b>Sine Component</b></td>
<td>
\[
\sin(z)
\]
</td>
<td><pre>sin_component = np.sin(z)</pre></td>
</tr>

<tr>
<td><b>Gaussian Component</b></td>
<td>
\[
\exp\!\left(-\frac{z^2}{32}\right)
\]
</td>
<td><pre>gauss_component = np.exp(-(z*z)/32)</pre></td>
</tr>

<tr>
<td><b>Model Output</b></td>
<td>
\[
\hat{y} = \sin(z)\,\exp\!\left(-\frac{z^2}{32}\right)
\]
</td>
<td><pre>y_pred = sin_component * gauss_component</pre></td>
</tr>

<!-- LOSS -->

<tr>
<td><b>Loss Function (SSE)</b></td>
<td>
\[
L = \sum_{i=1}^N (\hat{y}_i - y_i)^2
\]
</td>
<td><pre>loss = np.sum((pred_y - data_y)**2)</pre></td>
</tr>

<tr>
<td><b>Residual</b></td>
<td>
\[
e_i = \hat{y}_i - y_i
\]
</td>
<td><pre>e = pred_y - data_y</pre></td>
</tr>

<!-- GRADIENTS -->

<tr>
<td><b>∂ŷ/∂φ₀</b></td>
<td>
\[
\frac{\partial \hat{y}}{\partial \phi_0}
=
\cos(z)e^{-z^{2}/32}
-
\sin(z)e^{-z^{2}/32}\frac{z}{16}
\]
</td>
<td><pre>deriv = cos(z)*gauss - sin(z)*gauss*(z/16)</pre></td>
</tr>

<tr>
<td><b>∂L/∂φ₀</b></td>
<td>
\[
\frac{\partial L}{\partial \phi_0}
=
2\sum_i e_i
\left(
\frac{\partial \hat{y}_i}{\partial \phi_0}
\right)
\]
</td>
<td><pre>deriv = 2 * deriv * (sin_component*gauss_component - y)</pre></td>
</tr>

<tr>
<td><b>∂ŷ/∂φ₁</b></td>
<td>
\[
\frac{\partial \hat{y}}{\partial \phi_1}
=
0.06 x\,
\left[
\cos(z)e^{-z^{2}/32}
-
\sin(z)e^{-z^{2}/32}\frac{z}{16}
\right]
\]
</td>
<td><pre>deriv = 0.06*x*(cos(z)*gauss - sin(z)*gauss*(z/16))</pre></td>
</tr>

<tr>
<td><b>∂L/∂φ₁</b></td>
<td>
\[
\frac{\partial L}{\partial \phi_1}
=
2\sum_i e_i
\left(
\frac{\partial \hat{y}_i}{\partial \phi_1}
\right)
\]
</td>
<td><pre>deriv = 2*deriv*(sin_component*gauss_component - y)</pre></td>
</tr>

<tr>
<td><b>Gradient Vector</b></td>
<td>
\[
\nabla L(\phi)
=
\begin{bmatrix}
\frac{\partial L}{\partial \phi_0} \\
\frac{\partial L}{\partial \phi_1}
\end{bmatrix}
\]
</td>
<td><pre>gradient = np.array([[dl_dphi0],[dl_dphi1]])</pre></td>
</tr>

<!-- SGD UPDATE -->

<tr>
<td><b>Standard SGD Update</b></td>
<td>
\[
\phi \leftarrow \phi - \alpha \nabla L
\]
</td>
<td><pre>phi = phi - alpha * gradient</pre></td>
</tr>

<!-- MOMENTUM -->

<tr>
<td><b>Momentum Update (Equation 6.11)</b></td>
<td>
\[
v_t = \beta v_{t-1} + \nabla L(\phi_t)
\]
\[
\phi_{t+1} = \phi_t - \alpha v_t
\]
</td>
<td>
<pre>momentum = beta * momentum + gradient</pre>
<pre>phi = phi - alpha * momentum</pre>
</td>
</tr>

<!-- NESTEROV -->

<tr>
<td><b>Nesterov Lookahead</b></td>
<td>
\[
\tilde{\phi} = \phi_t - \beta v_{t}
\]
</td>
<td><pre>lookahead_phi = phi - beta * momentum</pre></td>
</tr>

<tr>
<td><b>Nesterov Gradient</b></td>
<td>
\[
\nabla L(\tilde{\phi})
\]
</td>
<td><pre>gradient = compute_gradient(..., lookahead_phi)</pre></td>
</tr>

<tr>
<td><b>Nesterov Momentum Update</b></td>
<td>
\[
v_t = \beta v_{t-1} + \nabla L(\tilde{\phi})
\]
</td>
<td><pre>momentum = beta * momentum + gradient</pre></td>
</tr>

<tr>
<td><b>Nesterov Parameter Update</b></td>
<td>
\[
\phi_{t+1} = \phi_t - \alpha v_t
\]
</td>
<td><pre>phi = phi - alpha * momentum</pre></td>
</tr>

</tbody>
</table>

FORMULAS FOR ADAM OPTIMIZER

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Operation</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Code Line (Exact Calculation)</th>
</tr>
</thead>
<tbody>

<!-- LOSS FUNCTION -->

<tr>
<td><b>Loss Function</b></td>
<td>
\[
L(\phi_0,\phi_1)
= 1 - 
\exp\!\left(-2\phi_1^2\right)
\exp\!\left(-\frac{(\phi_0 - 0.7)^2}{8}\right)
\]
</td>
<td>
<pre>height = exp(-0.5*(phi1*phi1)*4.0)*exp(-0.5*(phi0-0.7)**2/4.0)</pre>
</td>
</tr>

<!-- FINITE DIFFERENCE GRADIENT -->

<tr>
<td><b>Finite Difference Gradient (φ₀)</b></td>
<td>
\[
\frac{\partial L}{\partial \phi_0}
\approx
\frac{
L(\phi_0+\frac{\delta}{2},\phi_1)
-
L(\phi_0-\frac{\delta}{2},\phi_1)
}{\delta}
\]
</td>
<td>
<pre>gradient[0] = (loss(phi0+δ/2, φ1) - loss(phi0-δ/2, φ1)) / δ</pre>
</td>
</tr>

<tr>
<td><b>Finite Difference Gradient (φ₁)</b></td>
<td>
\[
\frac{\partial L}{\partial \phi_1}
\approx
\frac{
L(\phi_0,\phi_1+\frac{\delta}{2})
-
L(\phi_0,\phi_1-\frac{\delta}{2})
}{\delta}
\]
</td>
<td>
<pre>gradient[1] = (loss(phi0, φ1+δ/2) - loss(phi0, φ1-δ/2)) / δ</pre>
</td>
</tr>

<!-- STANDARD GRADIENT DESCENT -->

<tr>
<td><b>Standard Gradient Descent Update</b></td>
<td>
\[
\phi_{t+1}
=
\phi_{t}
-
\alpha \nabla L(\phi_t)
\]
</td>
<td>
<pre>grad_path[:,t+1] = grad_path[:,t] - α * grad</pre>
</td>
</tr>

<!-- NORMALIZED GRADIENT (ADAGRAD / RMSPROP STYLE) -->

<tr>
<td><b>Accumulated Squared Gradient</b></td>
<td>
\[
v_t = v_{t-1} + (\nabla L(\phi_t))^2
\]
</td>
<td>
<pre>v = v + m*m</pre>
</td>
</tr>

<tr>
<td><b>Normalized Gradient Update</b></td>
<td>
\[
\phi_{t+1}
=
\phi_t
-
\alpha
\frac{
\nabla L(\phi_t)
}{
\sqrt{v_t} + \epsilon
}
\]
</td>
<td>
<pre>grad_path[:,t+1] = grad_path[:,t] - α*(m/(sqrt(v)+ε))</pre>
</td>
</tr>

<!-- ADAM OPTIMIZER -->

<tr>
<td><b>Adam Momentum Estimate</b></td>
<td>
\[
m_t = \beta m_{t-1} + (1-\beta) \nabla L(\phi_t)
\]
</td>
<td>
<pre>m = β*m + (1-β)*grad</pre>
</td>
</tr>

<tr>
<td><b>Adam Variance Estimate</b></td>
<td>
\[
v_t = \gamma v_{t-1} + (1-\gamma)(\nabla L(\phi_t))^2
\]
</td>
<td>
<pre>v = γ*v + (1-γ)*grad*grad</pre>
</td>
</tr>

<tr>
<td><b>Bias-Corrected Momentum</b></td>
<td>
\[
\tilde{m}_t = \frac{m_t}{1-\beta^{t}}
\]
</td>
<td>
<pre>m_tilde = m / (1 - β^(t+1))</pre>
</td>
</tr>

<tr>
<td><b>Bias-Corrected Variance</b></td>
<td>
\[
\tilde{v}_t = \frac{v_t}{1-\gamma^{t}}
\]
</td>
<td>
<pre>v_tilde = v / (1 - γ^(t+1))</pre>
</td>
</tr>

<tr>
<td><b>Adam Parameter Update</b></td>
<td>
\[
\phi_{t+1}
=
\phi_t
-
\alpha
\frac{
\tilde{m}_t
}{
\sqrt{\tilde{v}_t} + \epsilon
}
\]
</td>
<td>
<pre>grad_path[:,t+1] = grad_path[:,t] - α*(m_tilde/(sqrt(v_tilde)+ε))</pre>
</td>
</tr>

</tbody>
</table>

FORMULAS FOR BACKPROPAGATION WITH TOY MODEL

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Quantity</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Exact Code Line</th>
</tr>
</thead>
<tbody>

<!-- FORWARD PASS -->

<tr>
<td><b>f₀</b></td>
<td>
\[
f_0 = \beta_0 + \omega_0 x
\]
</td>
<td><pre>f0 = beta0 + omega0 * x</pre></td>
</tr>

<tr>
<td><b>h₁ = sin(f₀)</b></td>
<td>
\[
h_1 = \sin(f_0)
\]
</td>
<td><pre>h1 = np.sin(f0)</pre></td>
</tr>

<tr>
<td><b>f₁</b></td>
<td>
\[
f_1 = \beta_1 + \omega_1 h_1
\]
</td>
<td><pre>f1 = beta1 + omega1 * h1</pre></td>
</tr>

<tr>
<td><b>h₂ = exp(f₁)</b></td>
<td>
\[
h_2 = \exp(f_1)
\]
</td>
<td><pre>h2 = np.exp(f1)</pre></td>
</tr>

<tr>
<td><b>f₂</b></td>
<td>
\[
f_2 = \beta_2 + \omega_2 h_2
\]
</td>
<td><pre>f2 = beta2 + omega2 * h2</pre></td>
</tr>

<tr>
<td><b>h₃ = cos(f₂)</b></td>
<td>
\[
h_3 = \cos(f_2)
\]
</td>
<td><pre>h3 = np.cos(f2)</pre></td>
</tr>

<tr>
<td><b>f₃</b></td>
<td>
\[
f_3 = \beta_3 + \omega_3 h_3
\]
</td>
<td><pre>f3 = beta3 + omega3 * h3</pre></td>
</tr>

<tr>
<td><b>Loss</b></td>
<td>
\[
L = (f_3 - y)^2
\]
</td>
<td><pre>l_i = (f3 - y)**2</pre></td>
</tr>

<!-- BACKWARD PASS -->

<tr>
<td><b>dL/df₃</b></td>
<td>
\[
\frac{\partial L}{\partial f_3}
= 2 (f_3 - y)
\]
</td>
<td><pre>dldf3 = 2 * (f3 - y)</pre></td>
</tr>

<tr>
<td><b>dL/dh₃</b></td>
<td>
\[
\frac{\partial L}{\partial h_3}
= \omega_3 \frac{\partial L}{\partial f_3}
\]
</td>
<td><pre>dldh3 = omega3 * dldf3</pre></td>
</tr>

<tr>
<td><b>dL/df₂</b></td>
<td>
\[
\frac{\partial L}{\partial f_2}
= -\sin(f_2)\,\frac{\partial L}{\partial h_3}
\]
</td>
<td><pre>dldf2 = (-np.sin(f2)) * dldh3</pre></td>
</tr>

<tr>
<td><b>dL/dh₂</b></td>
<td>
\[
\frac{\partial L}{\partial h_2}
= \omega_2 \frac{\partial L}{\partial f_2}
\]
</td>
<td><pre>dldh2 = omega2 * dldf2</pre></td>
</tr>

<tr>
<td><b>dL/df₁</b></td>
<td>
\[
\frac{\partial L}{\partial f_1}
= e^{f_1}\,\frac{\partial L}{\partial h_2}
\]
</td>
<td><pre>dldf1 = np.exp(f1) * dldh2</pre></td>
</tr>

<tr>
<td><b>dL/dh₁</b></td>
<td>
\[
\frac{\partial L}{\partial h_1}
= \omega_1 \frac{\partial L}{\partial f_1}
\]
</td>
<td><pre>dldh1 = omega1 * dldf1</pre></td>
</tr>

<tr>
<td><b>dL/df₀</b></td>
<td>
\[
\frac{\partial L}{\partial f_0}
= \cos(f_0)\,\frac{\partial L}{\partial h_1}
\]
</td>
<td><pre>dldf0 = np.cos(f0) * dldh1</pre></td>
</tr>

<!-- PARAMETER GRADIENTS -->

<tr>
<td><b>∂L/∂β₃</b></td>
<td>
\[
\frac{\partial L}{\partial \beta_3}
= \frac{\partial L}{\partial f_3}
\]
</td>
<td><pre>dldbeta3 = dldf3</pre></td>
</tr>

<tr>
<td><b>∂L/∂ω₃</b></td>
<td>
\[
\frac{\partial L}{\partial \omega_3}
= h_3 \frac{\partial L}{\partial f_3}
\]
</td>
<td><pre>dldomega3 = dldf3 * h3</pre></td>
</tr>

<tr>
<td><b>∂L/∂β₂</b></td>
<td>
\[
\frac{\partial L}{\partial \beta_2}
= \frac{\partial L}{\partial f_2}
\]
</td>
<td><pre>dldbeta2 = dldf2</pre></td>
</tr>

<tr>
<td><b>∂L/∂ω₂</b></td>
<td>
\[
\frac{\partial L}{\partial \omega_2}
= h_2 \frac{\partial L}{\partial f_2}
\]
</td>
<td><pre>dldomega2 = dldf2 * h2</pre></td>
</tr>

<tr>
<td><b>∂L/∂β₁</b></td>
<td>
\[
\frac{\partial L}{\partial \beta_1}
= \frac{\partial L}{\partial f_1}
\]
</td>
<td><pre>dldbeta1 = dldf1</pre></td>
</tr>

<tr>
<td><b>∂L/∂ω₁</b></td>
<td>
\[
\frac{\partial L}{\partial \omega_1}
= h_1 \frac{\partial L}{\partial f_1}
\]
</td>
<td><pre>dldomega1 = dldf1 * h1</pre></td>
</tr>

<tr>
<td><b>∂L/∂β₀</b></td>
<td>
\[
\frac{\partial L}{\partial \beta_0}
= \frac{\partial L}{\partial f_0}
\]
</td>
<td><pre>dldbeta0 = dldf0</pre></td>
</tr>

<tr>
<td><b>∂L/∂ω₀</b></td>
<td>
\[
\frac{\partial L}{\partial \omega_0}
= x \frac{\partial L}{\partial f_0}
\]
</td>
<td><pre>dldomega0 = dldf0 * x</pre></td>
</tr>

</tbody>
</table>

FORMULAS FOR BASIC BACKPROPAGATION

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Operation</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Exact Code Line</th>
</tr>
</thead>
<tbody>

<!-- SIGMOID -->

<tr>
<td><b>Sigmoid activation</b></td>
<td>
\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]
</td>
<td><pre>hidden_output = sigmoid(hidden_input)</pre></td>
</tr>

<tr>
<td><b>Sigmoid derivative</b></td>
<td>
\[
\sigma'(x)=x(1-x) \quad \text{(where x = sigmoid output)}
\]
</td>
<td><pre>d_output = error * sigmoid_derivative(final_output)</pre></td>
</tr>

<!-- FORWARD PASS -->

<tr>
<td><b>Hidden layer input</b></td>
<td>
\[
h_{\text{in}} = X W_1
\]
</td>
<td><pre>hidden_input = np.dot(X, W1)</pre></td>
</tr>

<tr>
<td><b>Hidden layer output</b></td>
<td>
\[
h_{\text{out}} = \sigma(h_{\text{in}})
\]
</td>
<td><pre>hidden_output = sigmoid(hidden_input)</pre></td>
</tr>

<tr>
<td><b>Output layer input</b></td>
<td>
\[
o_{\text{in}} = h_{\text{out}} W_2
\]
</td>
<td><pre>final_input = np.dot(hidden_output, W2)</pre></td>
</tr>

<tr>
<td><b>Final output</b></td>
<td>
\[
o_{\text{out}} = \sigma(o_{\text{in}})
\]
</td>
<td><pre>final_output = sigmoid(final_input)</pre></td>
</tr>

<!-- LOSS -->

<tr>
<td><b>Error (Loss)</b></td>
<td>
\[
E = y - o_{\text{out}}
\]
</td>
<td><pre>error = y - final_output</pre></td>
</tr>

<!-- BACKPROP -->

<tr>
<td><b>Output gradient</b></td>
<td>
\[
\delta_{\text{out}}
= (y - o_{\text{out}})\,\sigma'(o_{\text{out}})
\]
</td>
<td><pre>d_output = error * sigmoid_derivative(final_output)</pre></td>
</tr>

<tr>
<td><b>Hidden layer error</b></td>
<td>
\[
E_{\text{hidden}} = \delta_{\text{out}} W_2^\top
\]
</td>
<td><pre>error_hidden = d_output.dot(W2.T)</pre></td>
</tr>

<tr>
<td><b>Hidden layer gradient</b></td>
<td>
\[
\delta_{\text{hidden}} = E_{\text{hidden}} \,\sigma'(h_{\text{out}})
\]
</td>
<td><pre>d_hidden = error_hidden * sigmoid_derivative(hidden_output)</pre></td>
</tr>

<!-- WEIGHT UPDATES -->

<tr>
<td><b>Update W₂</b></td>
<td>
\[
W_2 \leftarrow W_2 + \eta\, h_{\text{out}}^\top \delta_{\text{out}}
\]
</td>
<td><pre>W2 += hidden_output.T.dot(d_output) * lr</pre></td>
</tr>

<tr>
<td><b>Update W₁</b></td>
<td>
\[
W_1 \leftarrow W_1 + \eta\, X^\top \delta_{\text{hidden}}
\]
</td>
<td><pre>W1 += X.T.dot(d_hidden) * lr</pre></td>
</tr>

</tbody>
</table>

FORMULAS FOR BACKPROPAGATION

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Backprop Equation</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Exact Code Line</th>
</tr>
</thead>
<tbody>

<!-- 7.22 Bias Gradient -->
<tr>
<td><b>Eq. 7.22 — Gradient wrt Bias<br>( ∂L/∂b )</b></td>
<td>
\[
\frac{\partial L}{\partial \boldsymbol{\beta}_k}
=
\frac{\partial L}{\partial \mathbf{f}_k}
\]
</td>
<td>
<pre>all_dl_dbiases[layer] = np.array(all_dl_df[layer])</pre>
</td>
</tr>

<!-- 7.23 Weight Gradient -->
<tr>
<td><b>Eq. 7.23 — Gradient wrt Weights<br>( ∂L/∂W )</b></td>
<td>
\[
\frac{\partial L}{\partial \boldsymbol{\Omega}_k}
=
\frac{\partial L}{\partial \mathbf{f}_k}\;
\mathbf{h}_{k}^{\top}
\]
</td>
<td>
<pre>all_dl_dweights[layer] =
    np.matmul(all_dl_df[layer], all_h[layer].T)</pre>
</td>
</tr>

<!-- 7.25 Backprop Through Layers -->
<tr>
<td><b>Eq. 7.25 — Gradient wrt Activation<br>( ∂L/∂h )</b></td>
<td>
\[
\frac{\partial L}{\partial \mathbf{h}_k}
=
\boldsymbol{\Omega}_k^{\top}
\frac{\partial L}{\partial \mathbf{f}_k}
\]
</td>
<td>
<pre>all_dl_dh[layer] =
    np.matmul(all_weights[layer].T, all_dl_df[layer])</pre>
</td>
</tr>

<tr>
<td><b>Eq. 7.25 — Gradient wrt Pre-activation<br>( ∂L/∂f ) via ReLU</b></td>
<td>
\[
\frac{\partial L}{\partial \mathbf{f}_k}
=
\frac{\partial L}{\partial \mathbf{h}_k}
\odot
\mathbf{1}_{\mathbf{f}_k > 0}
\]
</td>
<td>
<pre>all_dl_df[layer-1] =
    all_dl_dh[layer] * indicator_function(all_f[layer-1])</pre>
</td>
</tr>

<!-- ReLU derivative -->
<tr>
<td><b>ReLU Derivative</b></td>
<td>
\[
\frac{d}{df}\mathrm{ReLU}(f)
=
\begin{cases}
1 & f > 0 \\
0 & f \le 0
\end{cases}
\]
</td>
<td>
<pre>indicator_function(all_f[layer-1])</pre>
</td>
</tr>

</tbody>
</table>

FORMULAS FOR BACKPROPAGATION WITH PYTORCH

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Equation</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Code Implementation</th>
</tr>
</thead>
<tbody>

<!-- Pre-activation -->
<tr>
<td><b>Layer pre-activation</b></td>
<td>
\[
\mathbf{z}^{(k)} = \mathbf{a}^{(k-1)} W^{(k)} + \mathbf{b}^{(k)}
\]
</td>
<td>
<pre>z1 = x @ w1 + b1
z2 = a1 @ w2 + b2
z3 = a2 @ w3 + b3
y_pred = a3 @ w4 + b4</pre>
</td>
</tr>

<!-- Activation -->
<tr>
<td><b>Layer activation</b></td>
<td>
\[
\mathbf{a}^{(k)} = g(\mathbf{z}^{(k)})
\]
where  
\( g=\sin,\, \exp,\, \cos \)
</td>
<td>
<pre>a1 = torch.sin(z1)
a2 = torch.exp(torch.clamp(z2, max=4))
a3 = torch.cos(z3)</pre>
</td>
</tr>

<!-- Loss -->
<tr>
<td><b>Loss function</b></td>
<td>
\[
L = \frac{1}{N} \sum_i (\hat{y}_i - y_i)^2
\]
</td>
<td>
<pre>loss = loss_fn(y_pred, y)</pre>
</td>
</tr>

<!-- Backprop main chain -->
<tr>
<td><b>Backprop chain rule</b></td>
<td>
\[
\frac{\partial L}{\partial W^{(k)}} =
(\mathbf{a}^{(k-1)})^\top
\frac{\partial L}{\partial \mathbf{z}^{(k)}}
\]
<br>
\[
\frac{\partial L}{\partial b^{(k)}} =
\frac{\partial L}{\partial \mathbf{z}^{(k)}}
\]
</td>
<td>
<pre>loss.backward()</pre>
(PyTorch autograd computes all ∂L/∂W and ∂L/∂b)
</td>
</tr>

<!-- Activation derivative -->
<tr>
<td><b>Activation derivative</b></td>
<td>
For each layer:
\[
\frac{\partial L}{\partial z^{(k)}}
=
\frac{\partial L}{\partial a^{(k)}}
\odot g'(z^{(k)})
\]
<br><br>
Where:
\[
g'(z)=
\begin{cases}
\cos(z) & (\sin) \\
e^z & (\exp) \\
-\sin(z) & (\cos)
\end{cases}
\]
</td>
<td>
Handled automatically by autograd  
<pre>loss.backward()</pre>
</td>
</tr>

<!-- Weight update -->
<tr>
<td><b>Weight update rule (Adam)</b></td>
<td>
\[
\theta \leftarrow
\theta - \alpha \frac{m_t}{\sqrt{v_t}+\epsilon}
\]
(Adam update)
</td>
<td>
<pre>optimizer.step()</pre>
</td>
</tr>

<!-- Zero grad -->
<tr>
<td><b>Gradient reset</b></td>
<td>
\[
\nabla_{\theta} L = 0
\quad\text{(before next iteration)}
\]
</td>
<td>
<pre>optimizer.zero_grad()</pre>
</td>
</tr>

</tbody>
</table>

FORMULAS FOR BIAS VARIANCE 

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Concept</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Code Line</th>
</tr>
</thead>
<tbody>

<tr>
<td><b>True function</b></td>
<td>
\[
f(x)= e^{\sin(2\pi x)}
\]
</td>
<td><pre>def true_function(x):</pre></td>
</tr>

<tr>
<td><b>Noisy data</b></td>
<td>
\[
y_i = f(x_i) + \varepsilon_i,\qquad
\varepsilon_i \sim \mathcal{N}(0,\sigma_y^2)
\]
</td>
<td><pre>y[i] = true_function(x[i]) + np.random.normal(0, sigma_y)</pre></td>
</tr>

<tr>
<td><b>Hidden ReLU-like basis</b></td>
<td>
\[
h_j(x)=\max(0,\; x-\tfrac{j}{n_{\text{hid}}})
\]
</td>
<td><pre>h = line_vals * (line_vals > 0)</pre></td>
</tr>

<tr>
<td><b>Model output</b></td>
<td>
\[
\hat{y}(x)= \beta + 
\sum_{j=1}^{n_{\text{hid}}}
\omega_j\, h_j(x)
\]
</td>
<td><pre>y += omega[c_hidden] * h;  y += beta</pre></td>
</tr>

<tr>
<td><b>Closed-form least squares</b></td>
<td>
\[
\theta = (\mathbf{A}^\top \mathbf{A})^{-1}
\mathbf{A}^\top \mathbf{y}
\]
<br>
where  
\[
\theta = [\beta,\omega_1,\dots,\omega_H]
\]
</td>
<td><pre>beta_omega = np.linalg.lstsq(A, y)[0]</pre></td>
</tr>

<tr>
<td><b>Model mean</b></td>
<td>
\[
\mu(x)=\mathbb{E}[\hat{y}(x)]
\]
</td>
<td><pre>mean_model = np.mean(y_model_all, axis=0)</pre></td>
</tr>

<tr>
<td><b>Model variance</b></td>
<td>
\[
\mathrm{Var}(x)=
\mathbb{E}\!\left[(\hat{y}(x)-\mu(x))^2\right]
\]
</td>
<td><pre>std_model = np.std(y_model_all, axis=0)</pre></td>
</tr>

<tr>
<td><b>Variance (scalar)</b></td>
<td>
\[
\text{Variance} = 
\frac{1}{N_x}\sum_x \mathrm{Var}(x)
\]
</td>
<td><pre>variance[c_hidden] = np.mean(std_model**2)</pre></td>
</tr>

<tr>
<td><b>Squared bias</b></td>
<td>
\[
\text{Bias} =
\frac{1}{N_x}\sum_x
\bigl(\mu(x)-f(x)\bigr)^2
\]
</td>
<td><pre>bias[c_hidden] = np.mean((mean_model - y_func)**2)</pre></td>
</tr>

<tr>
<td><b>Total error decomposition</b></td>
<td>
\[
\text{Total} = \text{Bias} + \text{Variance}
\]
</td>
<td><pre>ax.plot(hidden_variables, variance+bias)</pre></td>
</tr>

</tbody>
</table>

FORMULAS FOR MNIST 1D PERFORMANCE

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Concept</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Code Line</th>
</tr>
</thead>
<tbody>

<tr>
<td><b>Input flattening</b></td>
<td>
\[
x \in \mathbb{R}^{784},\quad 
x_{\text{used}} = x_{0:40}
\]
</td>
<td><pre>images = images.view(... )[:, :D_i]</pre></td>
</tr>

<tr>
<td><b>Layer 1 pre-activation</b></td>
<td>
\[
f^{(1)} = W^{(1)} x + b^{(1)}
\]
</td>
<td><pre>nn.Linear(D_i, D_k)</pre></td>
</tr>

<tr>
<td><b>Layer 1 activation (ReLU)</b></td>
<td>
\[
h^{(1)} = \max(0, f^{(1)})
\]
</td>
<td><pre>nn.ReLU()</pre></td>
</tr>

<tr>
<td><b>Layer 2 pre-activation</b></td>
<td>
\[
f^{(2)} = W^{(2)} h^{(1)} + b^{(2)}
\]
</td>
<td><pre>nn.Linear(D_k, D_k)</pre></td>
</tr>

<tr>
<td><b>Layer 2 activation (ReLU)</b></td>
<td>
\[
h^{(2)} = \max(0, f^{(2)})
\]
</td>
<td><pre>nn.ReLU()</pre></td>
</tr>

<tr>
<td><b>Output layer</b></td>
<td>
\[
z = W^{(3)} h^{(2)} + b^{(3)} \in \mathbb{R}^{10}
\]
</td>
<td><pre>nn.Linear(D_k, D_o)</pre></td>
</tr>

<tr>
<td><b>Softmax (implicit, inside CrossEntropyLoss)</b></td>
<td>
\[
p_i = 
\frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
\]
</td>
<td><pre>criterion = nn.CrossEntropyLoss()</pre></td>
</tr>

<tr>
<td><b>Loss (Cross-Entropy)</b></td>
<td>
\[
\mathcal{L} = -\log p_{y}
\]
</td>
<td><pre>loss = criterion(outputs, labels)</pre></td>
</tr>

<tr>
<td><b>Gradient computation</b></td>
<td>
\[
W \leftarrow 
W - \eta \frac{\partial \mathcal{L}}{\partial W}
\]
</td>
<td><pre>loss.backward()</pre></td>
</tr>

<tr>
<td><b>Weight update (Adam)</b></td>
<td>
Adam update rules:
\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\]
\[
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\]
\[
\theta_t = 
\theta_{t-1}
- \alpha \frac{m_t}{\sqrt{v_t}+\epsilon}
\]
</td>
<td><pre>optimizer.step()</pre></td>
</tr>

<tr>
<td><b>Prediction</b></td>
<td>
\[
\hat{y} = \arg\max_i z_i
\]
</td>
<td><pre>_, predicted = torch.max(outputs.data, 1)</pre></td>
</tr>

<tr>
<td><b>Accuracy</b></td>
<td>
\[
\text{acc} = 
\frac{\text{correct}}{\text{total}}
\]
</td>
<td><pre>correct += (predicted == labels).sum()</pre></td>
</tr>

</tbody>
</table>

FORMULAS FOR DOUBLE DESCENT 

<table border="1" cellpadding="6" style="border-collapse: collapse; width:100%;">
<thead>
<tr>
<th>Concept</th>
<th>Mathematical Formula (LaTeX)</th>
<th>Code Line</th>
</tr>
</thead>
<tbody>

<tr>
<td><b>Input vector (first 40 dims)</b></td>
<td>
\[
x \in \mathbb{R}^{40}
\]
</td>
<td>
<pre>images = images.view(... )[:, :D_i]</pre>
</td>
</tr>

<tr>
<td><b>Layer 1 pre-activation</b></td>
<td>
\[
f^{(1)} = W^{(1)} x + b^{(1)}
\]
</td>
<td>
<pre>nn.Linear(D_i, D_k)</pre>
</td>
</tr>

<tr>
<td><b>Layer 1 activation (ReLU)</b></td>
<td>
\[
h^{(1)} = \max(0, f^{(1)})
\]
</td>
<td>
<pre>nn.ReLU()</pre>
</td>
</tr>

<tr>
<td><b>Layer 2 pre-activation</b></td>
<td>
\[
f^{(2)} = W^{(2)} h^{(1)} + b^{(2)}
\]
</td>
<td>
<pre>nn.Linear(D_k, D_k)</pre>
</td>
</tr>

<tr>
<td><b>Layer 2 activation (ReLU)</b></td>
<td>
\[
h^{(2)} = \max(0, f^{(2)})
\]
</td>
<td>
<pre>nn.ReLU()</pre>
</td>
</tr>

<tr>
<td><b>Output logits</b></td>
<td>
\[
z = W^{(3)} h^{(2)} + b^{(3)}
\]
</td>
<td>
<pre>nn.Linear(D_k, D_o)</pre>
</td>
</tr>

<tr>
<td><b>Softmax probabilities</b></td>
<td>
\[
p_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
\]
</td>
<td>
<pre>loss = nn.CrossEntropyLoss()</pre>
</td>
</tr>

<tr>
<td><b>Cross-entropy loss</b></td>
<td>
\[
\mathcal{L} = -\log(p_y)
\]
</td>
<td>
<pre>loss = loss_function(pred, y_batch)</pre>
</td>
</tr>

<tr>
<td><b>Loss gradient</b></td>
<td>
\[
g_t = \nabla_\theta \mathcal{L}
\]
</td>
<td>
<pre>loss.backward()</pre>
</td>
</tr>

<tr>
<td><b>SGD + Momentum</b></td>
<td>
\[
v_t = \mu v_{t-1} + g_t
\]
</td>
<td>
<pre>optimizer = SGD(..., momentum=0.9)</pre>
</td>
</tr>

<tr>
<td><b>Parameter update rule</b></td>
<td>
\[
\theta_{t+1} = \theta_t - \eta v_t
\]
</td>
<td>
<pre>optimizer.step()</pre>
</td>
</tr>

<tr>
<td><b>Prediction</b></td>
<td>
\[
\hat{y} = \arg\max_i z_i
\]
</td>
<td>
<pre>_, predicted = torch.max(outputs, 1)</pre>
</td>
</tr>

<tr>
<td><b>Accuracy</b></td>
<td>
\[
\text{Acc} = \frac{\text{Correct}}{N}
\]
</td>
<td>
<pre>(predicted == labels).sum()</pre>
</td>
</tr>

<tr>
<td><b>Error</b></td>
<td>
\[
\text{Err} = 1 - \text{Acc}
\]
</td>
<td>
<pre>test_err = 100 - accuracy</pre>
</td>
</tr>

<tr>
<td><b>Total trainable parameters</b></td>
<td>
\[
N_{\text{params}}
=
\sum_l \left(d^{\text{out}}_l d^{\text{in}}_l + d^{\text{out}}_l\right)
\]
</td>
<td>
<pre>count_parameters(model)</pre>
</td>
</tr>

<tr>
<td><b>Double Descent Critical Point</b></td>
<td>
\[
N_{\text{params}} = N_{\text{train}}
\]
</td>
<td>
<pre>closest_index = np.argmin(abs(W - N))</pre>
</td>
</tr>

</tbody>
</table>



