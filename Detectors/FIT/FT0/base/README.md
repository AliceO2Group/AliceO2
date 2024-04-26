# FT0 Base

## Geometry

The `o2::ft0::Geometry` class contains dimensions and information used for constructing the FT0 geometry as used in simulation. It also provides utility methods for retrieving the center locations of the detector cells. See the below example for how to use the `Geometry` class to query the FT0 cell locations. Note that these are the ideal locations, and that any misalignment is not considered.

```cpp
o2::ft0::Geometry ft0Geometry;
ft0Geometry.calculateChannelCenter();
TVector3 cellPosition = ft0Geometry.getChannelCenter(chId);
float x = cellPosition.X();
float y = cellPosition.Y();
float z = cellPosition.Z();
```
