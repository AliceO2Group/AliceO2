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
| `TRKBase.layoutMLOT` | `kCylindrical`, `kSegmented` (default), `kSimplifiedRealistic` | `kCylindrical`: simple silicon tubes. `kSegmented`: Turbo ML + solid-module OT. `kSimplifiedRealistic`: same ML as `kSegmented`, but a detailed OT barrel (see below) |
| `TRKBase.layoutSRV` | `kPeacockv1` (default), `kLOISymm` | `kLOISymm` produces radially symmetric service volumes, as used in the LoI |
| `TRKBase.otBarrelWallThickness` | thickness in cm (default `0.2`) | Carbon fibre separation walls of the OT quarter barrels — side panels and mid-rapidity disks (`kPeacockv1` + `kSimplifiedRealistic`); `0` disables them. Does not cover the load-bearing outer shell (4 mm) |
| `TRKBase.disableFT3` | `false` (default), `true` | toggle to disable the forward disks |
| `TRKBase.layoutFT3` | `kSegmentedStave` (default), `kSegmentedFT3`, `kTrapezoidal` | disk geometry settings `kSegmentedFT3` refers to an outdated segmentation |
| `TRKBase.nTrapezoidalSegments` | integer; default: 32 | number of trapezoidal segments in the disks for kTrapezoidal layout |


For example, a geometry with fully cylindrical tracker barrel (for all layers in VD, ML and OT) can be obtained by
```bash
o2-sim-serial-run5 -n 1 -g pythia8hi -m A3IP TRK TF3 \
  --configKeyValues "TRKBase.layoutVD=kIRISFullCyl;TRKBase.layoutMLOT=kCylindrical"
```

## Custom Geometry Configuration

The geometry of the ML and OT layers can be overridden by providing a custom plain-text configuration file via `TRKBase.configFile=filename.txt`. The parser interprets the file differently depending on the active `TRKBase.layoutMLOT` setting (`kCylindrical`, or `kSegmented`/`kSimplifiedRealistic` which share the same syntax).

### General Syntax Rules
* **Separators:** All columns **must** be separated by a single TAB (`\t`). Using spaces will result in a parsing error.
* **Comments:** Any line starting with a forward slash (`/`) is treated as a comment and ignored.
* **Layer Count:** The parser reads valid lines sequentially. The first valid line corresponds to Layer 0, the second to Layer 1, and so on.
* **Material Budget Mode:** All layer definitions accept an optional `matBudgetMode` parameter at the end of the line (e.g., `0` = Thickness, `1` = X2X0). If omitted, it defaults to `Thickness`.

### 1. Cylindrical Layout (`kCylindrical`)

When `TRKBase.layoutMLOT=kCylindrical` is used, each layer requires a minimum of 3 parameters to define the `TRKCylindricalLayer`.

* **Format:** `rInn` \t `length` \t `thick` \t `[optional_mode]`
* *(Note: `rInn`, `length`, and `thick` map directly to the constructor arguments for the cylindrical layer, typically corresponding to Radius, Length, and Thickness).*

**Example for `kCylindrical`:**
```text
/ Configuration for kCylindrical layout - ALICE3 TRK
/ rInn length thick [optional_mode]
7.0 127.985 0.1
9.0 127.985 0.1
12.0 127.985 0.1
20.0 127.985 0.1
30.0 127.985 0.1
45.0 255.9 0.1
60.0 255.9 0.1
80.0 255.9 0.1
```

### 2. Segmented / Simplified-Realistic Layout (`kSegmented`, `kSimplifiedRealistic`)

Both layouts use the same configuration-file syntax (only the OT geometry implementation differs). Each layer requires a minimum of 5 base parameters to define the geometry. The parser distinguishes between Middle Layers (ML) and Outer Layers (OT) based on the sequential layer index.

* *(Note: The 5 base parameters map directly to: Inner Radius (`rInn`), Thickness (`thick`), Tilt Angle (`tiltAngle`), Number of Staves (`nStaves`), and Number of Modules per stave (`nMods`)).*

**Middle Layers (ML) - Indices 0 to 4**
The first 5 valid lines are parsed as `TRKMLLayer` objects. These layers **require** a 6th parameter for the staggering offset (`stagOffset`).
* **Format:** `rInn` \t `thick` \t `tiltAngle` \t `nStaves` \t `nMods` \t `stagOffset` \t `[optional_mode]`

**Outer Layers (OT) - Indices 5 and above**
From the 6th valid line onwards, lines are parsed as OT layer objects (`TRKOTLayer` for `kSegmented`, `TRKOTLayerRealistic` for `kSimplifiedRealistic`). These layers do **not** have a staggering offset. The optional mode parameter shifts to the 6th column.
* **Format:** `rInn` \t `thick` \t `tiltAngle` \t `nStaves` \t `nMods` \t `[optional_mode]`
* *(Note: for `kSimplifiedRealistic`, `nStaves` is recomputed internally from the average radius and stave width to guarantee the neighbour overlap; the value in the file is ignored for the OT.)*

**Example for `kSegmented`:**

```text
/ Configuration for kSegmented layout - ALICE3 TRK
/ --- ML LAYERS (Indices 0 to 4) ---
/ rInn thick tilt nStaves nMods stagOffset [optional_mode]
7.0 0.01 11.2 10 11 0.0 1
9.0 0.01 11.9 14 11 0.0 1
12.0 0.01 11.4 18 11 0.0 1
20.0 0.01 0.0 26 11 1.17 1
30.0 0.01 0.0 38 11 0.89 1
/
/ --- OT LAYERS (Indices 5 to 7) ---
/ Outer layers do NOT have stagOffset.
/ rInn thick tilt nStaves nMods [optional_mode]
45.0 0.01 0.0 32 22 1
60.0 0.01 0.0 42 22 1
80.0 0.01 0.0 56 22 1
```

## Additional options for forward disks

Furthermore, there are more options in the case of stave segmentation -- for only OT or both. The user can set to cut the staves exactly on the nominal inner radii (true by default), and outer radii (false by default) of the disks. This exists since (planned) placements of sensors & staves often protrude out of the nominal radii to be more able to cover the nominal disk area. In addition, it is possible to draw reference circles (`TRKBase.drawReferenceCircles`) in root for the stave segmented layouts for both the inner (red) and outer (blue) radii. This is off by default, yet can be toggled if the user wants to see how tight the tiling is to the nominal radii -- for visualisation purposes only.

## Simplified-Realistic OT geometry (`kSimplifiedRealistic`)

`kSimplifiedRealistic` keeps the ML layers identical to `kSegmented` but replaces the solid-silicon OT modules with a more detailed, but still simplified, description (`TRKOTLayerRealistic`). It affects the **OT barrel only** — the forward disks are independent of `layoutMLOT` and are configured as described above. All tunable dimensions live in [`Specs.h`](./base/include/TRKBase/Specs.h) (`constants::OT`); values that depend on others are computed in the source.

The geometry specification is based on [ALICE3 OT WP1 Material (26.06.2025)](https://indico.cern.ch/event/1562183/contributions/6580808/attachments/3093672/5480049/ALICE3_OT_WP1_Material_260625.pdf).

**Module** — a flush stack about the chip mid-plane: cold plate (carbon fibre), 8 pure-silicon chips (2 in φ × 4 in z, dead zones facing the outer module edges), FPC (Kapton+Cu), one ZIF connector centred on a short edge, SMD capacitors over the chip footprints (skipping any under the connector), and two mounting brackets on the cold plate.

**Stave** — two module rows overlapping in φ (so each row's dead zone is covered by the other row's sensor) and offset in r by `rowRadialStagger`, plus a cooling pipe between them. The rows straddle the stave frame origin, so a stave placed on the barrel circle is tangent at its own centre: every row-to-row radial step in the barrel — inside a stave and between neighbours — is then the same `rowRadialStagger`, in every layer.

**End-of-stave card** — one readout PCB per stave (`constants::OT::eosCard`), mounted past the last module at the outer z end, coplanar with the two rows: a 120 × 80 mm, 1.5 mm board carrying four copper planes spread symmetrically through the thickness, the remainder FR4. The copper is what sets the card material budget: at the default 122 µm per plane the card is **4.0 % x/X₀** for a track crossing it perpendicularly (copper 3.40 %, FR4 0.60 %). It is tunable through `TRKBase.otEosCardCuThickness`, which changes only the copper/FR4 split inside the fixed 1.5 mm envelope, so the card dimensions and the layer envelopes stay put and only the radiation length moves. The cards reach |z| ≈ 141 cm, just short of the OT barrel service disk, which places them at |η| ≈ 1.84–1.88 on the innermost OT layer and |η| ≈ 1.27–1.32 on the outermost — inside the tracking acceptance, so they matter for forward-disk performance.

**Barrel** — each layer is built in **four parts**: two z-half-barrels (±η), each split azimuthally into two 180° halves. Both η half-barrels are cut on the **same vertical plane** (x = 0), so the region where the vertical beam-pipe supports run is free of staves over the whole barrel (the supports themselves are not in the geometry). Neighbouring staves overlap (≥ 1 mm active double-coverage); at the cut plane the two azimuthal halves instead leave a gap wide enough for the separation wall plus `barrelWallClearance` on each side. The stave count is the smallest even number that still meets the required overlap at the layer radius, so the OT radii (440, 615, 793 mm) are chosen at the top of a stave-count band, giving 30/42/54 staves per ring with 1.5-1.9 mm of overlap. The cooling pipe faces the larger radius on the inner two OT layers and the smaller radius on the flipped outer layer.

**Separation walls** — the quarter barrels are closed on every side but the one carrying the end-of-stave cards (`TRKServices::createOTBarrelWalls`, `kPeacockv1` + `kSimplifiedRealistic`): radially by the ML/OT and outer carbon shells, azimuthally by a rectangular carbon fibre wall in the cut plane, and at mid-rapidity by a half disk at z = 0, one per quarter barrel.

Both walls run **continuously** from one shell to the other: each OT layer envelope is a tube with a slot cut along the vertical cut plane *and* one at z = 0 (`TGeoCompositeShape`), so the walls pass through the layers instead of being interrupted by them, and what is left of each envelope is the four quarter barrels. The slots clear the walls by `barrelWallSlotMargin`; the staves stay `barrelWallClearance` away from them.

Only the outer shell is load bearing, so it is the only 4 mm wall; the ML/OT shell is 2 mm and the side panels and mid-rapidity disks default to 2 mm via `TRKBase.otBarrelWallThickness`.

The OT services (cables/cooling bundles, cold plates) are built by `TRKServices` for all layouts via `TRKBase.layoutSRV`.

<!-- doxy
/doxy -->
