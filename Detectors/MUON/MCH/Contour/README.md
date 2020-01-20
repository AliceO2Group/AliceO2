<!-- doxy
\page refDetectorsMUONMCHContour Contour Library
/doxy -->

# MCH Contour library

A header-only library to work with contours (polygons).

The main purpose of this library is to merge collection of polygons.
 Starting from polygons representing MCH pad, we can
 [construct](include/MCHContour/ContourCreator.h) polygons representing FEE
 (aka *motifs* in AliRoot parlance, or *padgroups*), PCBs
  or even full detection elements.

The `SVGWriter` class can be used to produce SVG representations of those
contours.
