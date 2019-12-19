<!-- doxy
\page refDetectorsBasetest Detectors Base test
/doxy -->

# WORK IN PROGRESS: Mat.Budget LUT classes


To generate the LUT (at the moment for R<400, with layers above 270 cm not optimized) run
```
root -b -q O2/Detectors/Base/test/buildMatBudLUT.C+
```

The generation is quite time consuming (may take ~30 min).

The optimized LUT will be stored in the matbud.root file.

Load it as:
```
auto mbr = o2::base::MatLayerCylSet::loadFromFile("matbud.root", "MatBud");
```

To query mat. budget between 2 points use:
```
float xyz0[3] = {0.,0.,0.};
float xyz1[3] = {70.,80.,20.};

auto mb = mbl.getMatBudget(xyz0[0],xyz0[1],xyz0[2], xyz1[0],xyz1[1],xyz1[2]);
// alternatively, use MatCell getMatBudget(const Point3D<float> &point0,const Point3D<float> &point1) method

std::cout << "<rho>= " << mb.meanRho << " <x/X0>= " << mb.meanX2X0 << "\n";
```
