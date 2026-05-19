<!-- doxy
\page refDetectorsUpgradesALICE3FT3 EndCaps
/doxy -->

# ALICE 3 Tracker Endcaps

This is top page for the FT3 detector documentation.

## Specific detector setup


Configuration of the endcap disks can be done by setting values for the `FT3Base.layoutFT3` configurable,
the available options are presented in the following Table:

| Option                            | Comments                                                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `kSegmentedStave`                 | Segmentation of ML and OT disks: Modules are placed on staggered staves with user defined constants               |
| `kSegmentedStaveOTOnly` (default) | Only OT disks are contain staves with modules, ML layers are segmented with strips of modules on front/back       |
| `kSegmented`                      | Segmentation of ML and OT disk with strips of modules of chips on the front and back of a layer                   |
| `kTrapezoidal`                    | Simple trapezoidal disks (in both ML and OT), with `FT3Base.nTrapezoidalSegments=32`                              |
| `kCylindrical`                    | Simplest possible disks as TGeoTubes (ML and OT), bad for ACTS (wrong digi due to polar coorinates on disk sides) |

Furthermore, there are more options in the case of stave segmentation -- for only OT or both. The user can set to cut the staves exactly on the nominal inner radii (true by default), and outer radii (false by default) of the disks. This exists since (planned) placements of sensors & staves often protrude out of the nominal radii to be more able to cover the nominal disk area. In addition, it is possible to draw reference circles in root for the stave segmented layouts for both the inner (red) and outer (blue) radii. This is off by default, yet can be toggled if the user wants to see how tight the tiling is to the nominal radii -- for visualisation purposes only.

[ [Link to definitions](./base/include/FT3Base/FT3BaseParam.h) ]

For example, see the command below to generate a geometry with the endcaps only, all layers with the stave geometry, and including reference circles of nominal radii for visualisation.
```bash
o2-sim-serial-run5 -n 1 -g pythia8hi -m FT3 \
  --configKeyValues "FT3Base.layoutFT3=kSegmented; FT3Base.drawReferenceCircles=true"
```

<!-- doxy
/doxy -->
