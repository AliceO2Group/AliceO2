# MCH Contour library

A header-only library to work with contours (polygons).

The main purpose of this library is to merge collection of polygons.
 Starting from polygons representing MCH pad, we can [construct](include/MCHContour/ContourCreator.h) polygons
  representing FEE (aka _motifs_ in AliRoot parlance, or _padgroups_), PCBs
  or even full detection elements.
  
The `SVGWriter` class can be used to produce SVG representations of those contours.
 