<!-- doxy
\page refDetectorsMUONMCHGeometryTest Test
/doxy -->

# MCH Geometry Helpers

The [MCHGeometryTest](./include/MCHGeometryTest/Helpers.h) library offers a few
utility functions to help debug the MCH geometry : [draw
it](./drawMCHGeometry.C), dump it as text, create a standalone version of it
(i.e. without other ALICE volumes)

Also provides a `getRadio` function to get a radiation length plot of a given
detection element.
