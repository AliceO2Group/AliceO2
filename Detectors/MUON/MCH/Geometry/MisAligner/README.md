<!-- doxy
\page refDetectorsMUONMCHGeometryMisAligner MisAlignment
/doxy -->

# MUON MCH Geometry MisAligner

This package hosts the functions to generate misalignments for MCH geometry

## MCH Zero MisAlignment

Zero mislaignment can be generated and copied to CCDB (or local file) :

```shell
root -b -q -e o2::mch::test::zeroMisAlignGeometry\(\)
```

## Generating MCH MisAlignment

MCH MisAlignments can be generated (and uploaded to CCDB or local file) using
the example root macro Geometry/Test/misAlign.C where one can set the values
for the generation of misalignments
