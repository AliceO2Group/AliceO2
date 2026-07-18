# DCAFitterN derivation notes

This file combines the extracted content of
`~/Downloads/DCAFitterN_derivation_for_codex.md` with the O2-specific covariance
updates made in `include/DCAFitter/DCAFitterN.h`.

## 1. Convention check from the extracted scan

The extracted scan starts with a matrix named `M_i` and states that it maps local
track coordinates to global coordinates:

```text
p_i^g = M_i p_i
```

but the printed 2D block

```text
[ cos(alpha_i)   sin(alpha_i) ]
[ -sin(alpha_i)  cos(alpha_i) ]
```

is the inverse of the O2 local-to-global rotation.  O2 uses

```text
R_i =
[ cos(alpha_i)  -sin(alpha_i)  0 ]
[ sin(alpha_i)   cos(alpha_i)  0 ]
[ 0              0             1 ]
```

with

```text
p_i^g = R_i p_i,        p_i = R_i^T p_i^g .
```

The later extracted formulas note this possible sign reversal.  The code in
`DCAFitterN.h` matches the O2 convention above.  In the rest of this document
`R_i` is always the O2 local-to-global rotation.

## 2. Weighted N-prong vertex fit

For track `i`, let the current point in its local frame be

```text
p_i = (x_i, y_i, z_i)^T .
```

Let `B_i` be the inverse covariance, or information matrix, for this local
point.  The fitted vertex `V` is in the global frame.  Its representation in
track `i`'s local frame is `R_i^T V`, so the residual is

```text
Delta_i = p_i - R_i^T V .
```

The weighted objective is

```text
chi2 = 1/2 sum_i Delta_i^T B_i Delta_i .
```

For fixed track points, differentiating with respect to `V` gives

```text
A V = sum_i R_i B_i p_i
```

where

```text
A = sum_i R_i B_i R_i^T .
```

Therefore

```text
V = A^{-1} sum_i R_i B_i p_i .
```

Define

```text
T_i = A^{-1} R_i B_i .
```

Then

```text
V = sum_i T_i p_i .
```

This is the `calcInverseWeight`, `calcPCACoefs`, and `calcPCA` structure in the
code.  The useful identity is

```text
sum_i T_i R_i^T = I .
```

## 3. Residuals after eliminating the vertex

Substituting `V = sum_k T_k p_k` into the residual gives

```text
Delta_i = sum_k D_ik p_k
```

with

```text
D_ik = delta_ik I - R_i^T T_k .
```

The fit parameters are the local running coordinates `x_k`.  During one Newton
linearization, `R_i`, `B_i`, `A`, and `T_i` are treated as fixed.  Each track
point depends only on its own parameter:

```text
d p_i / d x_k = delta_ik p_i'
```

so

```text
d Delta_i / d x_k = D_ik p_k'
```

and

```text
d^2 Delta_i / d x_k d x_l = delta_kl D_ik p_k'' .
```

## 4. Track derivatives in the O2 local frame

O2 central-barrel parameters are

```text
(Y, Z, snp, tgl, q/pt)
```

where

```text
snp = sin(phi_local),      csp = sqrt(1 - snp^2),
tgl = tan(lambda),         kappa = curvature = (q/pt) Bz B2C .
```

For the fast constant-`Bz` helix model:

```text
d snp / dX = kappa
dY / dX = snp / csp
dZ / dX = tgl / csp
```

and

```text
d2Y / dX2 = kappa / csp^3
d2Z / dX2 = kappa tgl snp / csp^3 .
```

Thus

```text
p_i'  = (1, dY/dX, dZ/dX)^T
p_i'' = (0, d2Y/dX2, d2Z/dX2)^T .
```

This matches `TrackDeriv::set`.

## 5. Gradient and Hessian

With symmetric `B_i`,

```text
g_k = d chi2 / d x_k
    = sum_i Delta_i^T B_i D_ik p_k' .
```

The exact Hessian is

```text
H_kl =
  sum_i (D_il p_l')^T B_i (D_ik p_k')
  + delta_kl sum_i Delta_i^T B_i D_ik p_k'' .
```

The first term is the Gauss-Newton term.  It contributes to diagonal and mixed
Hessian elements.  The second term is the residual-curvature term.  Since each
trajectory has an intrinsic second derivative only with respect to its own
running coordinate, this term contributes only when `k == l`.  This is the
reason for the code condition

```cpp
if (i == j) {
  ...
}
```

when computing the Hessian element `H_ij`.

The implementation solves

```text
H dX = g
```

and then applies

```text
X_new = X_old - dX .
```

This is equivalent to the more usual Newton notation `deltaX = -H^{-1} g`.

## 6. No-error special case

For the absolute-distance fit, take all information matrices as identity:

```text
B_i = I .
```

Then

```text
A = N I,       A^{-1} = (1/N) I,
T_i = (1/N) R_i .
```

The vertex is the average of global track points:

```text
V = (1/N) sum_i R_i p_i .
```

The residual is

```text
Delta_i = p_i - (1/N) sum_j R_i^T R_j p_j .
```

Define

```text
R_ij = (1/N) R_i^T R_j .
```

With the O2 convention,

```text
R_i^T R_j =
[ cos(ai-aj)   sin(ai-aj)  0 ]
[ -sin(ai-aj)  cos(ai-aj)  0 ]
[ 0            0           1 ] .
```

This matches the code definitions

```cpp
mCosDif[i][j] = (ci*cj + si*sj) / N;
mSinDif[i][j] = (si*cj - ci*sj) / N;
```

and the residual derivative components in `calcResidDerivativesNoErr`.

## 7. Track information matrix involving the local X axis

The old DCAFitterN code assigned a dummy uncertainty to local `X`, derived from
the local `Y` uncertainty.  The corrected treatment derives longitudinal vertex
information from the track geometry.

At fixed local `X`, the track measures `(Y,Z)` with covariance

```text
C_YZ =
[ C_YY  C_YZ ]
[ C_YZ  C_ZZ ] .
```

Let

```text
D = C_YY C_ZZ - C_YZ^2
```

and

```text
W = C_YZ^{-1}
  = 1/D [ C_ZZ   -C_YZ ]
        [ -C_YZ   C_YY ] .
```

Writing

```text
wYY =  C_ZZ / D
wYZ = -C_YZ / D
wZZ =  C_YY / D
```

and

```text
y' = dY/dX = snp/csp
z' = dZ/dX = tgl/csp
```

the vertex measurement matrix in local `(X,Y,Z)` coordinates is

```text
H =
[ -y'  1  0 ]
[ -z'  0  1 ] .
```

The local 3D information matrix is

```text
I = H^T W H .
```

Its independent elements are

```text
I_YY = wYY
I_YZ = wYZ
I_ZZ = wZZ
I_XY = -(wYY y' + wYZ z')
I_XZ = -(wYZ y' + wZZ z')
I_XX = y'^2 wYY + 2 y' z' wYZ + z'^2 wZZ .
```

These are the six members of `TrackCovI`:

```text
sxx, sxy, sxz, syy, syz, szz .
```

To combine tracks in the global frame, rotate the local information:

```text
I_i^g = R_i I_i R_i^T .
```

The vertex covariance is then

```text
C_V = (sum_i I_i^g)^{-1} .
```

## 8. Parent momentum covariance

The parent momentum is the sum of independent daughter momenta:

```text
P = sum_i p_i,        C_P = sum_i C_{p_i} .
```

For one O2 daughter track,

```text
px = pt (csp cos(alpha) - snp sin(alpha))
py = pt (snp cos(alpha) + csp sin(alpha))
pz = pt tgl
```

with `pt = |q|/|q/pt|`.  The charge is treated as exact and discrete.  The
derivatives with respect to native momentum parameters `(snp, tgl, q/pt)` are

```text
dpx/dsnp = -pt (snp cos(alpha)/csp + sin(alpha))
dpy/dsnp =  pt (cos(alpha) - snp sin(alpha)/csp)
dpz/dtgl =  pt

dpx/d(q/pt) = -px / (q/pt)
dpy/d(q/pt) = -py / (q/pt)
dpz/d(q/pt) = -pz / (q/pt)
```

and the omitted mixed derivatives in this Jacobian are zero:

```text
dpx/dtgl = 0,      dpy/dtgl = 0,
dpz/dsnp = 0 .
```

Let `A` be this 3 by 3 Jacobian and let `C_a` be the daughter covariance
submatrix in the native momentum-parameter order `(snp,tgl,q/pt)`.  Then

```text
C_p = A C_a A^T .
```

The six independent elements of `C_p` are accumulated into the O2 lab covariance
slots

```text
cov[9], cov[13], cov[14], cov[18], cov[19], cov[20]
```

which correspond to

```text
Cov(px,px), Cov(py,px), Cov(py,py),
Cov(pz,px), Cov(pz,py), Cov(pz,pz).
```

## 9. Parent track covariance

The final lab covariance passed to the O2 parent constructor contains the fitted
vertex covariance and the summed parent momentum covariance:

```text
C_lab =
[ C_V   0   ]
[ 0     C_P ] .
```

Position-momentum cross-covariances are not included in this approximation.
The existing O2 constructor

```cpp
TrackParCov(xyz, pxpypz, cov, charge, sectorAlpha)
```

then transforms this lab covariance to the selected parent track frame,
including the parent `alpha` convention.

## 10. Code changes summarized

1. `TrackCovI` now stores a full symmetric local 3D information matrix.
2. The dummy `XerrFactor` approximation was removed.
3. PCA weights and chi2 derivatives now use `sxx,sxy,sxz,syy,syz,szz`.
4. PCA covariance is the inverse of the summed global information matrix.
5. The Hessian residual-curvature term is restricted to diagonal Hessian
   elements.
6. `correctTracks()` now propagates the actual candidate track state with O2's
   analytic constant-`Bz` transport, keeping `mCandTr` and `mTrPos`
   synchronized.
7. `createParentTrackParCov()` now propagates daughter momentum covariance from
   native O2 `(snp,tgl,q/pt)` parameters to lab `(px,py,pz)` before constructing
   the parent `TrackParCov`.
