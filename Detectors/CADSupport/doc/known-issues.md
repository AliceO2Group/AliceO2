# Known issues

Open defects and limitations of `Detectors/CADSupport`. Work that has been decided on and deferred
is in `reference/Roadmap.md`; proposals that are not yet decided are in `ideas.md`.

## Performance

- `O2BVHSurfaceSolid` answers 4–5 % slower per query than the same code did before it moved into
  `libO2CADSupport`. About half of that arrives with the library boundary itself; the remainder is
  unattributed. No algorithm and no answer changed: this is measured on one part in four
  representations, with every per-kernel checksum identical.
- `o2-bench-cadsupport-xray` exits with status 1 when a run has lost crossings. It predates this
  module.
- `o2-bench-cadsupport-overlap --self-test` crashes. It predates this module.

## Correctness and robustness

- `O2BVHAssembly` builds its BVH and its bounding box lazily inside const queries, through
  `EnsureBuilt`, so two threads navigating a shape read from a file can race.
  `O2BVHSurfaceSolid` fills its caches in `CloseShape` and does not have this problem.
  `O2BVHAssembly` has no production caller today.
- `Detectors/Base`'s `O2Tessellated` switches its ray pruning off when the ray origin plus the root
  box exceeds `kMaxPruneScale` (about 2097 cm), and says nothing when it does.
- `O2OverlapCheck`'s containment-flip filter applies to `O2FlatCSG` samples only. Exact shapes keep
  the safety-band filter, because a probe along a concave edge slides along the neighbouring face.
- `Detectors/Base`'s `testMatBudLUT` fails in a development build because it looks for the TPC
  plugin in `lib` while the library is installed in `lib64`. It fails the same way on a clean `dev`.

## Documentation and tooling

- `validation/closure/roundtrip_module.sh` calls a Python interpreter through `$SW`, unlike the rest
  of the suite, which resolves its interpreter through `cadsupport.occ_env`.
- `cadsupport.occ_env` picks the first architecture holding pythonOCC when `O2_ROOT`'s own
  architecture has none.
