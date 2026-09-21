# Simulating ALICE geometries that come from CAD

Detectors are designed in CAD, but Geant transports particles through ROOT's TGeo geometry. This
guide is about crossing that gap automatically — taking an engineering model as it comes out of the
design office and turning it into something particles can be simulated through, all the way to hits
you can plot.

The usual way of crossing that gap is to read the drawings and write the geometry again by hand, in
C++, volume by volume. That works, and most of ALICE was built this way, but it is slow, it is easy
to get subtly wrong, and every time the engineers move a bracket the translation has to be redone.
For a detector that is still being designed — which is exactly the situation during an upgrade study
— the hand-written geometry is out of date almost as soon as it is written.

So instead we convert the CAD file directly. You export the assembly as STEP, run one converter over
it, and you get a ROOT macro that builds the geometry. From there a small JSON file tells `o2-sim` to
load that macro and place it in the ALICE world. Nothing is recompiled at any point, so the loop from
a new CAD revision to a new simulation takes minutes rather than weeks.

Getting the geometry in is only half of it, though. A shape that particles fly through is a passive
obstacle; to do physics you want it to *record* something. The second half of this guide is therefore
about the external-detector mechanism, which lets you declare parts of your imported geometry
sensitive and have them write hits — again with no detector class and no rebuild. That is usually
enough to answer the first questions an upgrade study asks: does this thing get hit, how often, and
where.

## What you will be able to do by the end

- Install the converter and check that it works.
- Convert a STEP assembly and look at the result.
- Understand and control how faithfully each part is represented.
- Attach materials, and know what the magnetic field and physics cuts will and will not do.
- Place the geometry inside ALICE as passive material.
- Make parts of it sensitive, run a simulation, and count hits.
- Take an existing ALICE detector out to CAD and back, and simulate the result.
- Know where the system's limits are, so you do not discover them in your results.

We assume you can run `o2-sim`, and nothing more. No CAD experience is needed, and no knowledge of
OpenCascade, which does the heavy lifting underneath but never has to be addressed directly.

## Where the code lives

Everything in this guide is in `Detectors/CADSupport` in [AliceO2](https://github.com/AliceO2Group/AliceO2).
`README.md` there is the complete option reference, and `doc/reference/` documents the solids, their
file formats and the recognition pipeline.

## Contents

**Start** — [Install the software](install.md) · [Convert your first model](first-conversion.md)

**Converting** — [How a part is represented](representation.md) ·
[Convert only part of a model](partial.md) · [Give it materials](materials.md) ·
[Field and cuts](field-and-cuts.md) · [The geom.C file](geom-c.md)

**Simulating** — [Add passive geometry](passive.md) · [Make it produce hits](hits.md) ·
[Grow it into a real detector](real-detector.md)

**Worked example** — [The ITS, out and back again](its-round-trip.md)

**Reference** — [Check your geometry](checks.md) · [Limits and pain points](limits.md)
