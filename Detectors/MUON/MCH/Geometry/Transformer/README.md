<!-- doxy
\page refDetectorsMUONMCHGeometryTransformer Transformations
/doxy -->

# MCH Geometry Transformations

Here you will find functions to perform transformations on MCH geometry : local
to global, global to local, and various (mis)alignement ones.

# MCH Transformations in JSON format

Also available is a CLI utility to extract MCH transformations (rotations and
translations) from a Root geometry file and export them in JSON format :

```shell
o2-mch-convert-geometry --geom o2sim_geometry.root > geom.json
```

The JSON output can be manipulated (like any other json file) with e.g. [jq](https://stedolan.github.io/jq/)

For instance, to sort the output per detection element id :

```shell
cat output.json | jq '.alignables|=sort_by(.deid)'
```

