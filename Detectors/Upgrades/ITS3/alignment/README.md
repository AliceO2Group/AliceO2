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
}```
