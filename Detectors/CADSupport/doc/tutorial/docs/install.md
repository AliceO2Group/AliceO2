# Install the software

The converter is a Python script, but it leans on OpenCascade — the CAD kernel that reads STEP files
— through its Python bindings, `pythonOCC`. That is the one piece you have to provide yourself.

> [!WARNING]
> **pythonOCC is not part of O2sim**
>
> It is a separate aliBuild package, and it is **not** pulled in when you build or load `O2sim`.
> If you have never built it, that is genuinely step one — no amount of loading `O2sim` will
> conjure it up.

So we build it first. This pulls in OpenCascade itself as a dependency, and takes a while the first
time:

```bash
cd ~/alisw
aliBuild build pythonOCC --defaults o2 --no-system SWIG
```

The `--no-system SWIG` is worth keeping even when aliBuild tells you the system SWIG will do. The
recipe asks for SWIG 4.2.1 and several distributions ship 4.2.0, which is close enough to be picked
up and not close enough to build. Forcing aliBuild to build its own costs a few minutes once and
saves a confusing failure later.

With that in place, everything happens in a single shell. We load `pythonOCC` together with `O2sim`,
because the converter needs ROOT as well as OpenCascade — and the same environment then runs `o2-sim`
afterwards, so there is no need to switch shells between converting and simulating:

```bash
alienv enter O2sim/latest,pythonOCC/latest
```

Two quick checks confirm the environment is sound. The first proves the CAD bindings import at all;
the second runs the converter's own self-test, which builds its test cases in memory and needs no
input file:

```bash
python3 -c "import OCC.Core.Bnd; print('OCC import OK')"
o2-cad-to-tgeo --self-test
```

```text
OCC import OK
...
20/20 in-field media checks passed
```

The two commands you will use throughout are `o2-cad-to-tgeo`, which takes STEP to TGeo, and
`o2-tgeo-to-cad`, which takes TGeo back to STEP. They are also installed under their older names,
`O2_CADtoTGeo.py` and `O2_TGeoToCAD.py`, which work identically.

> [!NOTE]
> **If the import fails with “No module named 'OCC'”**
>
> Some `pythonOCC` installations carry a modulefile that puts the `OCC` package directory itself on
> `PYTHONPATH`, rather than the `site-packages` directory containing it — so Python looks inside the
> package and never finds it. The cure is to drop the trailing `/OCC` from the
> `prepend-path PYTHONPATH` line in `$PYTHONOCC_ROOT/etc/modulefiles/pythonOCC`. A recipe fix is on
> its way to alidist.

## Outside the ALICE stack

A conda environment with `pythonocc-core` also works. There, run the script from the source tree:

```bash
conda create -n occ -c conda-forge python=3.10 pythonocc-core -y
conda activate occ
python3 $O2_SRC/Detectors/CADSupport/tools/O2_CADtoTGeo.py --help
```
