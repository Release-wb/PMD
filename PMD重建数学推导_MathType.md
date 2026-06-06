# 单目相位偏折术 PMD 重建数学推导

本文档给出单目相位偏折术（Phase Measuring Deflectometry, PMD）的核心数学推导。公式采用 MathType/LaTeX 兼容格式，便于复制到 Word、MathType 或 Markdown 编辑器中继续排版。

## 1. 坐标系定义

取相机坐标系作为全局坐标系。涉及坐标系如下：

| 符号 | 含义 |
|---|---|
| $I$ | 相机像素坐标系 |
| $C$ | 相机三维坐标系 |
| $S$ | 屏幕坐标系 |
| $B$ | 标定板坐标系 |
| $M$ | 参考镜面坐标系 |
| $R$ | 面型重建坐标系 |

相机中心记为：

$$
O_c =
\begin{bmatrix}
0 \\
0 \\
0
\end{bmatrix}
$$

相机内参矩阵为：

$$
K =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

屏幕坐标系到相机坐标系的变换为：

$$
X_c = R_s X_s + t_s
$$

其中，$R_s$ 为屏幕坐标系到相机坐标系的旋转矩阵，$t_s$ 为屏幕坐标系原点在相机坐标系下的位置。

参考镜面平面在相机坐标系下表示为：

$$
n_0^T X + d_0 = 0
$$

其中：

$$
n_0 =
\begin{bmatrix}
n_x \\
n_y \\
n_z
\end{bmatrix}
$$

为参考镜面单位法向量，$d_0$ 为平面参数。

## 2. 相机像素到空间射线

相机图像中的一个像素点为：

$$
p =
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}
$$

去畸变后，该像素对应相机坐标系中的归一化视线方向：

$$
r =
\frac{K^{-1}p}{\left\|K^{-1}p\right\|}
$$

因此，相机中心出发的空间射线可以写为：

$$
X(\lambda) = O_c + \lambda r
$$

因为相机中心取为坐标原点，所以：

$$
X(\lambda) = \lambda r
$$

## 3. 参考镜面平面标定

将标定板放在参考镜面上，或放在与参考镜面共面的安装位置。标定板坐标系中的点为：

$$
X_b =
\begin{bmatrix}
X_b \\
Y_b \\
0
\end{bmatrix}
$$

通过 PnP 可得到标定板坐标系到相机坐标系的变换：

$$
X_c = R_{cb}X_b + t_{cb}
$$

标定板平面在标定板坐标系中为：

$$
Z_b = 0
$$

其法向量为：

$$
n_b =
\begin{bmatrix}
0 \\
0 \\
1
\end{bmatrix}
$$

将法向量变换到相机坐标系：

$$
n_0 = R_{cb}n_b
$$

标定板坐标系原点在相机坐标系中的位置为：

$$
P_{0c} = t_{cb}
$$

因此参考镜面平面方程为：

$$
n_0^T(X - P_{0c}) = 0
$$

展开得：

$$
n_0^T X + d_0 = 0
$$

其中：

$$
d_0 = -n_0^T P_{0c}
$$

若标定板图案面与真实镜面反射面之间存在厚度偏移 $t$，需要沿法向补偿：

$$
P_m = P_{0c} + \alpha t n_0
$$

其中 $\alpha = 1$ 或 $\alpha = -1$ 由标定板法向方向和镜面所在侧决定。补偿后的镜面平面参数为：

$$
d_m = -n_0^T P_m
$$

补偿后的镜面平面方程为：

$$
n_0^T X + d_m = 0
$$

## 4. 相机射线与参考镜面求交

相机像素对应射线为：

$$
X(\lambda) = \lambda r
$$

参考镜面平面为：

$$
n_0^T X + d_0 = 0
$$

代入射线方程：

$$
n_0^T(\lambda r) + d_0 = 0
$$

解得：

$$
\lambda = -\frac{d_0}{n_0^T r}
$$

因此该像素对应的参考镜面反射点为：

$$
P_0 = \lambda r
$$

即：

$$
P_0 =
-\frac{d_0}{n_0^T r}r
$$

对于小面形误差的被测镜，通常采用参考平面近似：

$$
P \approx P_0
$$

其中，$P$ 为被测面上的实际反射点。

## 5. 屏幕坐标与相机坐标转换

屏幕像素坐标记为：

$$
(u_s, v_s)
$$

屏幕像素间距为 $p_x$ 和 $p_y$，屏幕坐标原点对应像素为 $(u_{s0}, v_{s0})$，则屏幕物理坐标为：

$$
X_s =
\begin{bmatrix}
(u_s - u_{s0})p_x \\
(v_s - v_{s0})p_y \\
0
\end{bmatrix}
$$

屏幕点在相机坐标系下为：

$$
Q = R_s X_s + t_s
$$

其中 $Q$ 是相机坐标系中的屏幕三维点。

## 6. 四步相移相位解算

屏幕显示 $x$ 方向正弦条纹：

$$
I_k^x = A + B\cos(\phi_x + \delta_k)
$$

屏幕显示 $y$ 方向正弦条纹：

$$
I_k^y = A + B\cos(\phi_y + \delta_k)
$$

四步相移量为：

$$
\delta_1 = 0
$$

$$
\delta_2 = \frac{\pi}{2}
$$

$$
\delta_3 = \pi
$$

$$
\delta_4 = \frac{3\pi}{2}
$$

对应采集强度为：

$$
\begin{aligned}
I_1 &= A + B\cos\phi \\
I_2 &= A + B\cos\left(\phi + \frac{\pi}{2}\right) \\
I_3 &= A + B\cos(\phi + \pi) \\
I_4 &= A + B\cos\left(\phi + \frac{3\pi}{2}\right)
\end{aligned}
$$

包裹相位为：

$$
\phi =
\operatorname{atan2}(I_4 - I_2,\ I_1 - I_3)
$$

因此：

$$
\phi_x =
\operatorname{atan2}(I_4^x - I_2^x,\ I_1^x - I_3^x)
$$

$$
\phi_y =
\operatorname{atan2}(I_4^y - I_2^y,\ I_1^y - I_3^y)
$$

包裹相位范围为：

$$
\phi_x,\phi_y \in (-\pi,\pi]
$$

经过相位展开后得到绝对相位：

$$
\Phi_x = \operatorname{unwrap}(\phi_x)
$$

$$
\Phi_y = \operatorname{unwrap}(\phi_y)
$$

## 7. 绝对相位到屏幕坐标

设屏幕条纹周期为 $T_x$ 和 $T_y$，单位为屏幕像素。若不考虑初始相位偏置，则：

$$
u_s = \frac{T_x}{2\pi}\Phi_x
$$

$$
v_s = \frac{T_y}{2\pi}\Phi_y
$$

若存在初始相位偏置 $\Phi_{x0}$ 和 $\Phi_{y0}$，则：

$$
u_s = \frac{T_x}{2\pi}(\Phi_x - \Phi_{x0})
$$

$$
v_s = \frac{T_y}{2\pi}(\Phi_y - \Phi_{y0})
$$

因此，通过相位解算可以建立映射：

$$
(u,v) \longrightarrow (u_s,v_s)
$$

即相机像素到屏幕像素的对应关系。

## 8. 参考面相位与被测面相位

分别采集参考平面镜和被测镜的相移图像，得到参考面绝对相位：

$$
\Phi_{\text{ref},x}(u,v)
$$

$$
\Phi_{\text{ref},y}(u,v)
$$

被测面绝对相位：

$$
\Phi_{\text{obj},x}(u,v)
$$

$$
\Phi_{\text{obj},y}(u,v)
$$

相位差为：

$$
\Delta\Phi_x =
\Phi_{\text{obj},x} -
\Phi_{\text{ref},x}
$$

$$
\Delta\Phi_y =
\Phi_{\text{obj},y} -
\Phi_{\text{ref},y}
$$

对应屏幕像素偏移：

$$
\Delta u_s =
\frac{T_x}{2\pi}\Delta\Phi_x
$$

$$
\Delta v_s =
\frac{T_y}{2\pi}\Delta\Phi_y
$$

因此被测面对应的屏幕坐标为：

$$
u_{\text{obj}} =
u_{\text{ref}} + \Delta u_s
$$

$$
v_{\text{obj}} =
v_{\text{ref}} + \Delta v_s
$$

再由屏幕标定关系得到三维屏幕点：

$$
Q_{\text{obj}} =
R_s X_{s,\text{obj}} + t_s
$$

参考面对应屏幕点为：

$$
Q_{\text{ref}} =
R_s X_{s,\text{ref}} + t_s
$$

## 9. 反射定律求镜面法向

对每一个相机像素，已知：

$$
O_c
$$

$$
P
$$

$$
Q
$$

其中 $O_c$ 为相机中心，$P$ 为镜面反射点，$Q$ 为屏幕点。

镜面点指向相机中心的单位向量为：

$$
v_c =
\frac{O_c - P}{\left\|O_c - P\right\|}
$$

镜面点指向屏幕点的单位向量为：

$$
v_s =
\frac{Q - P}{\left\|Q - P\right\|}
$$

根据镜面反射定律，镜面法向为入射方向与反射方向的角平分线：

$$
n =
\frac{v_c + v_s}{\left\|v_c + v_s\right\|}
$$

参考平面法向为：

$$
n_{\text{ref}} =
\frac{v_c + v_{\text{ref}}}
{\left\|v_c + v_{\text{ref}}\right\|}
$$

其中：

$$
v_{\text{ref}} =
\frac{Q_{\text{ref}} - P}
{\left\|Q_{\text{ref}} - P\right\|}
$$

被测面法向为：

$$
n_{\text{obj}} =
\frac{v_c + v_{\text{obj}}}
{\left\|v_c + v_{\text{obj}}\right\|}
$$

其中：

$$
v_{\text{obj}} =
\frac{Q_{\text{obj}} - P}
{\left\|Q_{\text{obj}} - P\right\|}
$$

## 10. 法向转斜率

设被测面相对参考平面的高度为：

$$
z = h(x,y)
$$

则其法向量与梯度关系为：

$$
n \parallel
\begin{bmatrix}
-\frac{\partial h}{\partial x} \\
-\frac{\partial h}{\partial y} \\
1
\end{bmatrix}
$$

设：

$$
n =
\begin{bmatrix}
n_x \\
n_y \\
n_z
\end{bmatrix}
$$

则：

$$
\frac{\partial h}{\partial x}
=
-\frac{n_x}{n_z}
$$

$$
\frac{\partial h}{\partial y}
=
-\frac{n_y}{n_z}
$$

参考面斜率为：

$$
p_{\text{ref}} =
-\frac{n_{\text{ref},x}}{n_{\text{ref},z}}
$$

$$
q_{\text{ref}} =
-\frac{n_{\text{ref},y}}{n_{\text{ref},z}}
$$

被测面斜率为：

$$
p_{\text{obj}} =
-\frac{n_{\text{obj},x}}{n_{\text{obj},z}}
$$

$$
q_{\text{obj}} =
-\frac{n_{\text{obj},y}}{n_{\text{obj},z}}
$$

相对斜率为：

$$
p =
p_{\text{obj}} - p_{\text{ref}}
$$

$$
q =
q_{\text{obj}} - q_{\text{ref}}
$$

因此：

$$
\frac{\partial h}{\partial x} = p
$$

$$
\frac{\partial h}{\partial y} = q
$$

## 11. 斜率积分恢复面形

已知梯度场：

$$
h_x = p
$$

$$
h_y = q
$$

即：

$$
\frac{\partial h}{\partial x} = p
$$

$$
\frac{\partial h}{\partial y} = q
$$

两式分别对 $x$ 和 $y$ 求导：

$$
\frac{\partial^2 h}{\partial x^2}
=
\frac{\partial p}{\partial x}
$$

$$
\frac{\partial^2 h}{\partial y^2}
=
\frac{\partial q}{\partial y}
$$

相加得到 Poisson 方程：

$$
\frac{\partial^2 h}{\partial x^2}
+
\frac{\partial^2 h}{\partial y^2}
=
\frac{\partial p}{\partial x}
+
\frac{\partial q}{\partial y}
$$

即：

$$
\nabla^2 h =
\operatorname{div}(g)
$$

其中：

$$
g =
\begin{bmatrix}
p \\
q
\end{bmatrix}
$$

$$
\operatorname{div}(g)
=
\frac{\partial p}{\partial x}
+
\frac{\partial q}{\partial y}
$$

因此面型重建问题转化为求解：

$$
\nabla^2 h =
\frac{\partial p}{\partial x}
+
\frac{\partial q}{\partial y}
$$

## 12. DCT-Poisson 积分形式

令：

$$
f =
\frac{\partial p}{\partial x}
+
\frac{\partial q}{\partial y}
$$

则：

$$
\nabla^2 h = f
$$

对 $f$ 做二维 DCT：

$$
\hat{f}_{mn} = \operatorname{DCT2}(f)
$$

Poisson 方程在频域中可写为：

$$
\lambda_{mn}\hat{h}_{mn} = \hat{f}_{mn}
$$

其中：

$$
\lambda_{mn}
=
\frac{2\cos\left(\frac{\pi m}{M}\right)-2}{\Delta y^2}
+
\frac{2\cos\left(\frac{\pi n}{N}\right)-2}{\Delta x^2}
$$

于是：

$$
\hat{h}_{mn}
=
\frac{\hat{f}_{mn}}{\lambda_{mn}}
$$

对于零频项：

$$
\lambda_{00} = 0
$$

因此高度的绝对常数项不可恢复，通常令：

$$
\hat{h}_{00} = 0
$$

最后通过反 DCT 得到高度：

$$
h = \operatorname{IDCT2}(\hat{h})
$$

由于梯度积分无法确定绝对高度零点，通常去除 piston：

$$
h \leftarrow h - \overline{h}
$$

必要时还可以去除 tip/tilt：

$$
h \leftarrow h - (ax + by + c)
$$

## 13. 小角度近似公式

在小斜率、近轴、屏幕与镜面近似平行的条件下，局部斜率会导致屏幕反射点产生位移。

设屏幕到镜面的距离为 $L$，则有近似关系：

$$
\Delta x_s \approx 2L\frac{\partial h}{\partial x}
$$

$$
\Delta y_s \approx 2L\frac{\partial h}{\partial y}
$$

因此：

$$
\frac{\partial h}{\partial x}
\approx
\frac{\Delta x_s}{2L}
$$

$$
\frac{\partial h}{\partial y}
\approx
\frac{\Delta y_s}{2L}
$$

屏幕物理位移由相位差给出：

$$
\Delta x_s
=
p_x\Delta u_s
=
p_x\frac{T_x}{2\pi}\Delta\Phi_x
$$

$$
\Delta y_s
=
p_y\Delta v_s
=
p_y\frac{T_y}{2\pi}\Delta\Phi_y
$$

代入可得：

$$
\frac{\partial h}{\partial x}
\approx
\frac{p_xT_x}{4\pi L}\Delta\Phi_x
$$

$$
\frac{\partial h}{\partial y}
\approx
\frac{p_yT_y}{4\pi L}\Delta\Phi_y
$$

该公式直观，但仅适用于简化几何。高精度 PMD 应使用完整的 $C$-$P$-$Q$ 反射几何求法向。

## 14. 完整重建链路总结

单目 PMD 的完整数学链路为：

$$
(u,v)
\xrightarrow{K}
r
\xrightarrow{n_0,d_0}
P
\xrightarrow{\Phi_x,\Phi_y}
Q
\xrightarrow{\text{reflection}}
n
\xrightarrow{}
(p,q)
\xrightarrow{\text{integration}}
h(x,y)
$$

更具体地：

$$
\begin{aligned}
(u,v)
&\rightarrow r = \frac{K^{-1}[u,v,1]^T}{\left\|K^{-1}[u,v,1]^T\right\|} \\
&\rightarrow P = -\frac{d_0}{n_0^T r}r \\
&\rightarrow (u_s,v_s) =
\left(
\frac{T_x}{2\pi}\Phi_x,
\frac{T_y}{2\pi}\Phi_y
\right) \\
&\rightarrow Q = R_sX_s + t_s \\
&\rightarrow n =
\frac{
\frac{O_c-P}{\left\|O_c-P\right\|}
+
\frac{Q-P}{\left\|Q-P\right\|}
}
{
\left\|
\frac{O_c-P}{\left\|O_c-P\right\|}
+
\frac{Q-P}{\left\|Q-P\right\|}
\right\|
} \\
&\rightarrow
p = -\frac{n_x}{n_z},
\quad
q = -\frac{n_y}{n_z} \\
&\rightarrow
\nabla^2 h =
\frac{\partial p}{\partial x}
+
\frac{\partial q}{\partial y}
\end{aligned}
$$

## 15. 三个关键标定量

单目 PMD 定量重建必须获得三个核心几何量：

### 15.1 相机内参

由圆点板或棋盘格多姿态相机标定获得：

$$
K,\quad \text{distortion}
$$

其作用是：

$$
(u,v) \rightarrow r
$$

### 15.2 参考镜面平面

由标定板放在镜面上，通过 PnP 获得：

$$
n_0,\quad d_0
$$

其作用是：

$$
r \rightarrow P
$$

### 15.3 屏幕位姿

由屏幕显示圆点阵列，直接拍摄或通过参考镜反射拍摄获得：

$$
R_s,\quad t_s
$$

其作用是：

$$
(u_s,v_s) \rightarrow Q
$$

三者缺一不可。没有相机标定，就不能得到像素射线；没有参考镜面标定，就不能得到反射点；没有屏幕标定，就不能得到屏幕三维点。

