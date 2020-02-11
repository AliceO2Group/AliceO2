<!-- doxy
\page refDetectorsMUONMCHMappingFactory Factory (helper function)
/doxy -->

A single utility function to create segmentations, avoiding duplications
of objects.

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

Note that depending on your use case, both approaches might be valid ones.
While the factory one ensures you get only get one object for each detection
element, it creates all detection elements at once. So, if you only need the
segmentation of one detection element, you'd better off *not* using the
factory.
