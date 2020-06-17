<!-- doxy
\page refDetectorsMUONMCHMapping Mapping
* \subpage refDetectorsMUONMCHMappingImpl3 
* \subpage refDetectorsMUONMCHMappingImpl4
* \subpage refDetectorsMUONMCHMappingtest
* \subpage refDetectorsMUONMCHMappingSegContour
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

## The segmentation() utility function

To create segmentations and avoid duplicating objects, you can use the `segmentation`
 function (mind the lower case first letter s).

For instance :

```cpp
const Segmentation& seg1 = segmentation(501);
...
(some code here)
...
auto& seg2 = segmentation(501);
```

will result in only one object for segmentation 501 to be created.

On the contrary, if you use :

```cpp
Segmentation seg1{501};
...
...
Segmentation seg2{501};
```

two separate objects will be created (and you will be hit twice by the
object creation time).

> Note that depending on your use case, *both* approaches might be valid ones !
> While the factory one ensures you only get one object for each detection
> element, it creates all detection elements at once. So, if you only need the
> segmentation of one detection element, you'd better off *not* using the
> factory.

## Implementations

Currently two implementations ([Impl3](Impl3/README.md) and
[Impl4](Impl4/README.md), with Impl4 being the default one) are provided.  Both
are faster than the legacy AliRoot one (but not by an order of magnitude...).
Extensive checks have been made to insure that Impl3 yields the same results as
AliRoot (see the `vsaliroot` directory in the [alo
repository](https://github.com/mrrtf/alo) ).  The only difference between the
two implementations are the channel numbering. Impl3 uses Manu channel number,
aka Run1-2 numbering, while the Impl4 uses DualSampa channel numbering, aka
Run3 numbering. Impl3 is to be considered legacy and is used to deal with Run2
data only. It will be deprecated and removed at some point.

Note that there is some (high) level of duplication between the two libraries,
but that's by design as the two libraries are meant to be completely
independent : by construction they can *not* be used at the same time, and at
some point in the future we'll remove Impl3 for instance. The only commonality
they have is that they implement the same interface, but they should not share
implementation details.

## Contours

Besides to core API (finding pads by position and by electronic ids), we offer
[convenience functions](SegContour/README.md) to compute the bounding box and [envelop of the segmentations](SegContour/include/MCHMappingSegContour/SegmentationContours.h).

A simple executable, `o2-mch-mapping-svg-segmentation3` can be used to produce [SVG](https://developer.mozilla.org/en-US/docs/Web/SVG) plots
of the segmentations.

## Tests

A collection of [tests](test/README.md) (with or without input) are available.

## Command line interface (CLI)

A very basic CLI is provided for each implementation :  `o2-mch-mapping-cli3` or
 `o2-mch-mapping-cli4`.
