# Segmentation contours

This module contains convenience functions that can compute the
 envelop (in the form of a polygon) of segmentations.
 
Also an executable is provided which can produce SVG representation
 of those segmentations.
 
```c++
> mch-mapping-svg-segmentation3 --help
Generic options:
  --help                produce help message
  --hidepads            hide pad outlines
  --hidedualsampas      hide dualsampa outlines
  --hidedes             hide detection element outline
  --hidepadchannels     hide pad channel numbering
  --de arg              which detection element to consider
  --prefix arg (=seg)   prefix used for outfile filename(s)
  --point arg           points to show
  --all                 use all detection elements
```

Example usage :

```c++
> mch-mapping-svg-segmentation3 --hidepadchannels --hidepads --de 100 --prefix chamber1
```

Will produce an HTML file showing the FEE boundaries of [one quadrant of Station 1](chamber1-100-B.html)


