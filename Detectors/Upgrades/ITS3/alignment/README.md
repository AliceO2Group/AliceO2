# Simulate ITS3 misalignment and re-alignment


```bash
o2-its3-alignment-workflow --track-sources ITS --output MilleData,MilleSteer --configKeyValues "ITS3AlignmentParams.minPt=0.1;ITS3AlignmentParams.doMisalignmentLeg=true;ITS3AlignmentParams.doMisalignmentRB=true;ITS3AlignmentParams.misAlgJson=test_closure.json;ITS3AlignmentParams.extraClsErrZ[0]=10e-4;ITS3AlignmentParams.extraClsErrY[0]=10e-4;ITS3AlignmentParams.extraClsErrZ[3]=10e-4;ITS3AlignmentParams.extraClsErrY[3]=10e-4;ITS3AlignmentParams.dofConfigJson=dofSet.json" -b --run
```

test_closure.json:
```json
[
  {
    "id": 0,
    "rigidBody": [0.001, 0.0005, 0.0, 0.0, 0.0001, 0.0],
    "matrix": [[0.0], [0.0008, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
  }
]
```

dofSet.json:
```json
{
  "defaults": { "rigidBody": "fixed" },
  "rules": [
    {
      "match": "ITS3Layer0/ITS3CarbonForm0",
      "rigidBody": ["TX", "TY", "RY"],
      "calib": { "type": "legendre", "order": 1, "fix": [0, 2] }
    }
  ]
}
```


## In-extensional modes

The deformation of the open half-shell is parameterised by two 1D functions expanded in Legendre polynomials of the
normalised azimuth `u` (the same coordinate as the radial Legendre model):

```
f(phi) = sum_k f_k P_k(u),   g(phi) = sum_k g_k P_k(u)
u_z = f,   u_phi = -(z/r) f' + g,   u_r = (z/r) f'' - g'
```

`order` sets the maximum `k`. Optionally, strictly radial ("extensional") modes can be added on top, `u_r += sum_{k,l}
h_{k,l} P_k(u) P_l(v)` with `l >= 1`, enabled via `extOrderPhi` (max `k`) and `extOrderZ` (max `l`); `l = 0` is excluded
because a z-independent radial field is already spanned by the `g` family.

Note that `f_0` (translation along the cylinder axis) and `g_0` (rotation about it) are rigid-body motions and are fixed
by default; free them only if the rigid-body DOFs of the same volume are not fitted.

```json
{
  "defaults": { "rigidBody": "fixed" },
  "rules": [
    {
      "match": "ITS3Layer1/ITS3CarbonForm0",
      "calib": {
        "type": "inextensional",
        "order": 10,
        "extOrderPhi": 7,
        "extOrderZ": 8,
        "fix": ["f_0", "g_0"]
      }
    }
  ]
}
```

Injected/fitted coefficients (`h` keys are `"<k>_<l>"`):

```json
[
  {
    "id": 2,
    "inextensional": {
      "f": { "1": 0.0001, "2": -0.0002 },
      "g": { "1": 0.0625, "3": 0.0335, "5": -0.0453 },
      "h": { "4_2": -0.0421, "6_2": 0.0252, "4_4": 0.0435 }
    }
  }
]
```
