# ZDC Fast Simulation

The fast simulation replaces the Geant transport in ZN and ZP with a generative neural network (ONNX).
For each spectator nucleon the model directly gives the number of photons in the 10 hadronic channels:

```
ZN: common, ch1, ch2, ch3, ch4  |  ZP: common, ch1, ch2, ch3, ch4
```

The output is stored as standard `o2::zdc::Hit`, so digitization and reconstruction are unchanged.

## Enable it

```bash
o2-sim -m ZDC ... --configKeyValues "ZDCSimParam.useZDCFastSim=true"
```

| `ZDCSimParam.` | Default | |
|---|---|---|
| `useZDCFastSim` | `false` | enable the fast simulation |
| `fastSimCCDBPath` | `ZDC/Config/FastSimModel` | CCDB path of the model |
| `fastSimLocalFile` | `""` | local ROOT file with the model (overrides the CCDB) |

With the fast simulation enabled, the ZDC geometry is not built and the cave is the short one.

## How it works

At the end of each (sub-)event, `Detector::finishEventHook()` receives the primaries and:

1. keeps the primaries inside the model validity domain (`FastSimModel::accepts()`: PDG, energy, |η|, vertex);
2. mirrors in z the particles with `pz < 0` (the model is trained on the A side only);
3. runs the model on all of them in one batch, with the noise drawn from `o2::base::simRandom()` (reproducible with the seed);
4. creates one hit per primary and channel on the A or C side.
   The common channel is stored as PMC light, the towers as PMQ light.

## Create the model object

```bash
root -b -q 'CreateFastSimModel.C("zdc_fastsim.onnx", 0, -1, "local", "v1")'
```

The macro is in `Detectors/ZDC/macro`.
It packs the `.onnx` file together with its tensor names and validity cuts into an `o2::zdc::FastSimModel`.
It writes `ZDCFastSimModel.root`; pass `"test"`, `"internal"`, `"external"` or a URL instead of `"local"` to upload to the CCDB.
The default cuts accept everything: set them to the training domain.

Expected ONNX signature (dynamic batch `N`, float):

| Tensor | Shape | Content |
|---|---|---|
| `cond_phys` | `[N,8]` | PDG, E, vx, vy, vz, px, py, pz (GeV, cm, raw values) |
| `x0` | `[N,10]` | gaussian noise |
| `u` (optional) | `[N,10]` | uniform numbers for the stochastic rounding |
| `x_phys` | `[N,10]` | photons per channel |

Preprocessing and the full sampling procedure must be inside the graph.

## Limitations

- Only primaries accepted by the model give a signal (no fragments, no secondaries).
- No ZEM signal.
- No spatial information (`recordSpatialResponse` has no effect).
