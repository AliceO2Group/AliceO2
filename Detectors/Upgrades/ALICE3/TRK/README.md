<!-- doxy
\page refDetectorsUpgradesALICE3TRK Tracker
/doxy -->

# ALICE 3 Tracker Barrel

This is top page for the TRK detector documentation.


## Specific detector setup


Configurables for various sub-detectors are presented in the following Table:

| Subsystem          | Available options                                       | Comments                                                         |
| ------------------ | ------------------------------------------------------- | ---------------------------------------------------------------- |
| `TRKBase.layoutVD` | `kIRIS4` (default), `kIRISFullCyl`, `kIRIS5`, `kIRIS4a` | [link to definitions](./base/include/TRKBase/TRKBaseParam.h) |
| `TRKBase.layoutMLOT` | `kCylindrical`, `kSegmented` (default)     | `kSegmented` produced a Turbo layout for ML and a Staggered layout for OT                                                                 |
| `TRKBase.layoutSRV` | `kPeacockv1` (default), `kLOISymm` | `kLOISymm` produces radially symmetric service volumes, as used in the LoI |

For example, a geometry with fully cylindrical tracker barrel (for all layers in VD, ML and OT) can be obtained by
```bash
o2-sim-serial-run5 -n 1 -g pythia8hi -m A3IP TRK FT3 TF3 \
  --configKeyValues "TRKBase.layoutVD=kIRISFullCyl;TRKBase.layoutMLOT=kCylindrical"
```

<!-- doxy
/doxy -->
