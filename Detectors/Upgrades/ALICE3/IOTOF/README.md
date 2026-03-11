<!-- doxy
\page refDetectorsUpgradesALICE3IOTOF TOF
/doxy -->

# ALICE 3 TOF system

This is top page for the TOF detector documentation.


## Specific detector setup


Configurables for various sub-detectors are presented in the following Table:

[link to definitions](./base/include/IOTOFBase/IOTOFBaseParam.h)

| Options                       | Choices                                                          | Comments                                       |
| ----------------------------- | ---------------------------------------------------------------- | ---------------------------------------------- |
| `IOTOFBase.enableInnerTOF`    | `true` (default), `false`                                        | Enable inner TOF barrel layer                  |
| `IOTOFBase.enableOuterTOF`    | `true` (default), `false`                                        | Enable outer TOF barrel layer                  |
| `IOTOFBase.enableForwardTOF`  | `true` (default), `false`                                        | Enable forward TOF endcap                      |
| `IOTOFBase.enableBackwardTOF` | `true` (default), `false`                                        | Enable backward TOF endcap                     |
| `IOTOFBase.segmentedInnerTOF` | `false` (default), `true`                                        | Use segmented geometry for inner TOF           |
| `IOTOFBase.segmentedOuterTOF` | `false` (default), `true`                                        | Use segmented geometry for outer TOF           |
| `IOTOFBase.detectorPattern`   | ` ` (default), `v3b`, `v3b1a`, `v3b1b`, `v3b2a`, `v3b2b`, `v3b3` | Optional layout pattern                        |
| `IOTOFBase.x2x0`              | `0.02` (default)                                                 | Chip thickness in fractions of the rad. lenght |


For example, a geometry with fully cylindrical tracker barrel (for all layers in VD, ML and OT) can be obtained by
```bash
o2-sim-serial-run5 -n 1 -g pythia8hi -m A3IP  TF3 \
  --configKeyValues "IOTOFBase.detectorPattern=v3b1a;IOTOFBase.segmentedInnerTOF=true;IOTOFBase.segmentedOuterTOF=true;FT3Base.geoModel=1;FT3Base.nLayers=1;IOTOFBase.enableOuterTOF=false;IOTOFBase.enableBackwardTOF=false;IOTOFBase.enableForwardTOF=false;"
```

<!-- doxy
/doxy -->
