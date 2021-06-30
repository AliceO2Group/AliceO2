<!-- doxy
\page refDetectorsMUONMCHGeometryTransformer Transformations
/doxy -->

# MCH Geometry Transformations

Here you will find functions to perform transformations on MCH geometry : local
to global, global to local, and various (mis)alignement ones.

## MCH Transformations in JSON format

Also available is a CLI utility to extract MCH transformations (rotations and
translations) from a Root geometry file and export them in JSON format :

```shell
o2-mch-convert-geometry --geom o2sim_geometry.root > geom.json
```

The JSON output is compact on purpose, to save some space (e.g. on github).

But as any JSON it can be manipulated further with e.g. [jq](https://stedolan.github.io/jq/)

For instance you can pretty-print it :

```shell
cat output.json | jq .
```

Or sort the output per detection element id :

```shell
cat output.json | jq '.alignables|=sort_by(.deid)'
```

That last command being the one that was used to produce the example [ideal-geometry-o2.json](../Test/ideal-geometry-o2.json) file.

