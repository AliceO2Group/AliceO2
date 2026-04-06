<!-- doxy
\page refDetectorsUpgradesALICE3FT3 EndCaps
/doxy -->

# ALICE 3 Tracker Endcaps

This is top page for the FT3 detector documentation.

## Specific detector setup


Configuration of the endcap disks can be done by setting values for the `FT3Base.layoutFT3` configurable,
the available options are presented in the following Table:

| Option                 | Comments                                                                                                          |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `kSegmented` (default) | Currently, only OT disks have realistic implementation, for ML - simple trapezoids                                |
| `kTrapezoidal`         | Simple trapezoisal disks (in both ML and OT), with `FT3Base.nTrapezoidalSegments=32`                              |
| `kCylindrical`         | Simplest possible disks as TGeoTubes (ML and OT), bad for ACTS (wrong digi due to polar coorinates on disk sides) |

[ [Link to definitions](./base/include/FT3Base/FT3BaseParam.h) ]

For example, a geometry with the endcaps-only can be obtained by
```bash
o2-sim-serial-run5 -n 1 -g pythia8hi -m FT3 \
  --configKeyValues "FT3Base.layoutFT3=kTrapezoidal"
```

<!-- doxy
/doxy -->
