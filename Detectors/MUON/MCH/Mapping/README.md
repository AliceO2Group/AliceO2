<!-- doxy
\page refDetectorsMUONMCHMapping Mapping
* \subpage refDetectorsMUONMCHMappingImpl3 
* \subpage refDetectorsMUONMCHMappingtest
* \subpage refDetectorsMUONMCHMappingSegContour
* \subpage refDetectorsMUONMCHMappingFactory
/doxy -->

# MCH Mapping Interface

Two APIs are offered to the MCH Mapping :
a [C interface](Interface/include/MCHMappingInterface/SegmentationCInterface.h)
(the core one) and a [C++ interface](Interface/include/MCHMappingInterface/Segmentation.h).

The pattern of this dual interface follows the [Hourglass idea](https://github.com/CppCon/CppCon2014/tree/master/Presentations/Hourglass%20Interfaces%20for%20C%2B%2B%20APIs).
In a nutshell, the core interface is written in C so it can be accessed easily
from different languages and offers a stable ABI.

But most users are only exposed to a header-only [C++
interface](Interface/include/MCHMappingInterface/Segmentation.h) which sits on
top of this core C interface.

Details are to be found in the interface file or in the doxygen documentation,
but the basics of the interface is :

```c++
int detElemId{100};

o2::mch::mapping::Segmentation seg{detElemId};

double x{1.5};
double y{18.6};

int b, nb;
book found = seg.findPadPairByPosition(x, y, b, nb);

if (seg.isValid(b)) {
std::cout << "There is a bending pad at position " << x << "," << y << "\n"
<< " which belongs to dualSampa " << seg.padDualSampaId(b)
<< " and has a x-size of " << seg.padSizeX(b) << " cm\n";
}

assert(b == seg.findPadByFEE(76, 9));
```

Note that cathode-specific segmentations can be retrieved from `Segmentation`
objects (using bending() and nonBending() methods) or created from scratch
using the `CathodeSegmentation(int detElemId, bool isBendingPlane)` constructor.

## Implementations

Currently [one implementation](Impl3/README.md) is provided. It is faster than
the legacy AliRoot one (but not by an order of magnitude...). Extensive checks
have been made to insure it yields the same results as AliRoot (see the
`vsaliroot` directory in the [alo repository](https://github.com/mrrtf/alo) ).

## Contours

Besides to core API (finding pads by position and by electronic ids), we offer
[convenience functions](SegContour/README.md) to compute the bounding box and [envelop of the segmentations](SegContour/include/MCHMappingSegContour/SegmentationContours.h).

A simple executable, `o2-mch-mapping-svg-segmentation3` can be used to produce [SVG](https://developer.mozilla.org/en-US/docs/Web/SVG) plots
of the segmentations.

## Tests

A collection of [tests](test/README.md) (with or without input) are available.
