<!-- doxy
\page refDetectorsMUONMCHMappingImpl4 Impl4
/doxy -->

# MCH Mapping Implementation

This is an implementation of the [C
interface](../Interface/include/MCHMappingInterface/SegmentationCInterface.h)
of the MCH mapping.  The C function themselves are implemented using a [C++
class](src/SegmentationImpl4.h).  (which is not to be confused with the [C++
interface](../Interface/include/MCHMappingInterface/Segmentation.h)).

This implementation is the first complete version of the original idea
 to move to a self-contained mapping, i.e. mapping fully contained into
 code, without requiring external files (either ASCII or OCDB files).

Note that most of the files here (the `Gen*` ones) have been *generated* 
with [alo/jsonmap/codegen](https://github.com/mrrtf/alo/tree/master/jsonmap/codegen)
from a [JSON](https://www.json.org) representation of the mapping.

Internally we use a data structure meant for spatial searching : a `R-tree`
(from the [Boost Geometry
Index](http://www.boost.org/doc/libs/1_66_0/libs/geometry/doc/html/geometry/spatial_indexes/introduction.html)
library) to store and query the pads.
