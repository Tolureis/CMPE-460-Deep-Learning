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

