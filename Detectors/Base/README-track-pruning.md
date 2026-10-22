# ONNX transport pruning: simnet.birth.v1

O2 supplies one float32 `[batch,34]` input named `birth_features_v1`. The order is
`TrackTransportFeatureNames` in `TrackTransportUtils.h`, mirrored by
`TRANSPORT_FEATURES` in simnet's `src/training/qa_tools.py`:

```
charge_sign,mass,ekin,pt,eta,phi,theta,vx,vy,vz,r_xy,t_ns,
pdg_class,lifetime_ns,bending_radius_cm,z_at_r40,z_at_r250,medium_code,
has_gen_mother,pdg,abs_pdg,energy,px,py,pz,p,rapidity,dx_from_event,
dy_from_event,dz_from_event,r_from_event_xy,r3_from_event,mother_pdg,mother_abs_pdg
```

These are birth-time quantities, never hit labels or post-transport bookkeeping.
`has_gen_mother` uses the first mother, including transport mothers. Missing
ancestry, event vertex or medium is represented by NaN in the affected columns.
Birth-medium lookup uses a private per-thread navigator.

The model uses ONNX Gather to select/reorder its configured `labels_x` columns.
Only selected inputs are checked for finite values and the model's training
ranges. Unused NaN/Inf columns have no effect; multiplying them by zero would
not be sufficient. Scaling is embedded after selection, including any external
NN standard scaler. Invalid selected inputs return a NaN score, which O2 always
interprets as KEEP, including when inversion is enabled.

There is one float32 `[batch,1]` output, `probability_hit_free_subtree`. The NN's
sigmoid and the BDT's selection of probability column 1 are inside the graph.
Class 1 means no recorded detector hits in the track's entire descendant tree.
It does not mean electrically neutral or prove zero deposited energy.

Configuration:

```
Stack.transportPrimary=onnx
Stack.transportPrimaryOnnxCCDBPath=<raw ONNX object path>
Stack.transportPrimaryOnnxThreshold=<saved validation-selected threshold>
Stack.transportPrimaryOnnxOutputIndex=0
Stack.transportPrimaryOnnxApplySigmoid=false
Stack.transportPrimaryInvert=false
Stack.transportPrimaryOnnxSecondaries=false
```

The threshold has double precision to preserve cuts immediately above tied
float32 scores. The default -1 requires explicit configuration. The threshold
`nextafter(1,+inf)` is also accepted to represent a keep-all operating point.
CCDB validity and creation-time constraints must cover the simulation timestamp.

Pruning defaults to simulation roots, including injected tracks, matching the
current simnet training configs. Enable secondary pruning only after training
and validating that cohort. Feature selection does not change the training
population or make existing weights suitable for unseen populations.

With the updated training scripts, deploy `network/net.onnx` or
`network/bdt.onnx`. `*.core.onnx` files are intermediate models with selected
inputs, not O2-compatible exports. Earlier 19/21/25-input exports are rejected.
The full input/selection contract and operating point are saved beside the
model as `net.json` or `bdt.json` and embedded in ONNX metadata.
