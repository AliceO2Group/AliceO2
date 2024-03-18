# FV0 Base

## Geometry

The `o2::fv0::Geometry` class represents the geometry of the FV0 detector as used in simulation. It also provides utility methods for retrieving the center locations of the detector cells. See the below example for how to use the `Geometry` class to query the FV0 cell locations. Note that these are the ideal locations, and that any misalignment is not considered.

```cpp
o2::fv0::Geometry* fv0Geometry = o2::fv0::Geometry::instance(o2::fv0::Geometry::eUninitialized);
o2::fv0::Point3Dsimple cellPosition = fv0Geometry->getReadoutCenter(chId);
float x = cellPosition.x;
float y = cellPosition.y;
float z = cellPosition.z;
```
