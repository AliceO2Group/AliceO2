# ACTS tracking geometry in O2

Provides the shared, immutable `Acts::TrackingGeometry` that ACTS-based
reconstruction needs, built from the TGeo geometry O2 already has live in memory.

ACTS is an **optional** O2 dependency (`dependencies/O2Dependencies.cmake`). Without
it this package produces no targets and `O2_WITH_ACTS` is not defined, so anything
using it must be guarded.

## Libraries

| Target | Contents |
| --- | --- |
| `O2::ACTSInterface` | `TrackingGeometryManager` (the provider), `ITrackingGeometryBuilder` (the detector-specific seam), `SurfaceIndexMap` (sensor ↔ surface lookup), `MagneticFieldAdapter` (`Acts::MagneticFieldProvider` over the O2 field). No DPL dependency. |
| `O2::ACTSWorkflow` | `ActsGeometryService`, the same provider reachable through the DPL `ServiceRegistry`. |
| `O2::ALICE3ACTS` | `Gen3BlueprintBuilder`, the ALICE 3 builder (under `Detectors/Upgrades/ALICE3/ACTS`). |

## Using it from a DPL task

The manager fetches nothing itself: it needs `gGeoManager` to be live, which is what
`GRPGeomRequest::Aligned` arranges. Note that a task that only used the material LUT
before will have been passing `GRPGeomRequest::None` and has to be switched.

```cpp
// spec factory
auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(
    false, false, false, true /*GRPMagField*/, false,
    o2::base::GRPGeomRequest::Aligned, inputs, true);

void init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBReq);

  o2::alice3::Gen3BlueprintBuilder::Config cfg;
  cfg.geometryConfigFile = ic.options().get<std::string>("acts-geometry-config");
  o2::acts::TrackingGeometryManager::instance().setBuilder(
      std::make_unique<o2::alice3::Gen3BlueprintBuilder>(cfg));
}

void run(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);      // gGeoManager now live
  const auto& tg = o2::acts::TrackingGeometryManager::instance().get();  // built once
}
```

To reach it through the registry instead, put `o2::acts::defaultServicesWithActsGeometry()`
into the `DataProcessorSpec`'s `requiredServices` and use
`pc.services().get<o2::acts::ActsGeometryService>()`. Both paths hand out the same object.

## Mapping O2 clusters onto ACTS surfaces

`makeSurfaceIndex()` (or `TrackingGeometryManager::getIndex(cache, tol, pathProvider)`,
which caches per detector) maps every sensor of a `DetMatrixCache`-derived geometry helper
onto a sensitive ACTS surface.

**Pass a `SensorPathProvider` whenever the detector has one.** With it, sensors are
identified by the TGeo node the surface was built from — an identity, so it cannot
mis-assign:

```cpp
auto* trkGeo = o2::trk::GeometryTGeo::Instance();
trkGeo->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));
const auto& index = mgr.getIndex(*trkGeo, 1e-3,
    [trkGeo](int chipID) { return std::string(trkGeo->getMatrixPath(chipID).Data()); });
```

Without it the fallback matches on the sensor's local-to-global transform, which is exact
where a transform identifies a sensor but not always: the ALICE 3 vertex-detector petals
are tube segments, so the three layers of a petal share both origin and rotation and
differ only in radius, which the matrix cache does not carry. Those are reported as
ambiguous rather than assigned arbitrarily, and `makeSurfaceIndex()` throws. One surface
per sensor is enforced in both modes.

## The Gen3 geometry description

`Detectors/Upgrades/ALICE3/ACTS/config/gen3_geometry_config.json` carries everything
geometry-specific: sensor name globs, region boundaries, per-subsystem clustering
tolerances, passive material structures and the pinned volume IDs. All lengths in mm.

It is tuned for the layout `run_test.sh` generates — verified against an o2-sim geometry of
that layout: the sensor globs cover every sensitive volume family (and correctly exclude
`FT3Sensor_Inactive_*`), `*EOSCard*` matches the 504 end-of-stave volumes, the passive
cylinder radii and half-lengths match the real support volumes, and the resulting
volume/layer/surface table is identical to the one actsO2's reference geometry produces.

One known imprecision, inherited and present for both geometries: `TRK_MID_CarbonSupport`
is declared with `halfZ = 1420` while the real `TRK_MID_CARBONSUPPORT` volume has
`dz = 1410 mm`. It only affects the z-extent of that passive layer's material receiver.

## Material

Gen3 construction takes no `IMaterialDecorator`, so an ACTS JSON material map is applied
afterwards via `setMaterialMapFile()`. Material maps are keyed on
`Acts::GeometryIdentifier`, so a structural change to a builder invalidates existing
maps; the manager raises rather than silently producing a material-free geometry.

## Checking a geometry

`Detectors/Upgrades/ALICE3/ACTS/macros/run_test.sh` does the whole thing: it runs
`o2-sim-serial-run5` with the ALICE 3 layout the ACTS chain is developed against, then
builds the tracking geometry from the result and checks every TRK chip against its ACTS
surface. Run it from a scratch directory — `o2-sim` writes into `$PWD`:

```bash
mkdir -p /tmp/acts && cd /tmp/acts
bash <O2-source>/Detectors/Upgrades/ALICE3/ACTS/macros/run_test.sh
```

The geometry description it needs is shipped in this package
(`Detectors/Upgrades/ALICE3/ACTS/config/gen3_geometry_config.json`, installed to
`$O2_ROOT/share/Detectors/Upgrades/ALICE3/ACTS/config/`) and matches the layout the script
generates. Set `GEN3_CONFIG` to point at a different geometry's config.

Knobs: `nEvents`, `generator`, `modules`, `GEN3_CONFIG`, `SKIP_ALIGNMENT`. The last one
defaults to 1 because `o2-sim` finishes by fetching alignment from CCDB, which needs a
valid alien token and aborts without one — after the geometry file has already been
written. The ACTS geometry is built from the ideal geometry, so skipping it costs nothing.

The macro `CheckActsTrackingGeometry.C` can also be run on its own against an existing
`o2sim_geometry.root`; it prints the volume/layer/surface table, which is the authority for
the `(volume, layer)` keys that digitisation and seeding configurations use. It needs ACTS,
Eigen and `$O2_ROOT/include` on `ROOT_INCLUDE_PATH` — `run_test.sh` sets that up.
