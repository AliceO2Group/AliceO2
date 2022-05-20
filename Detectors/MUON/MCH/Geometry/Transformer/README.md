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

## Workflow

The `o2-mch-clusters-transformer-workflow` takes as input the list of all clusters ([Cluster](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)), in local reference frame, in the current time frame, with the data description "CLUSTERS".

It sends the list of the same clusters, but converted in global reference frame, with the data description "GLOBALCLUSTERS".

By default the workflow reads the (aligned) geometry from the CCDB.
 One can instead read the geometry from a file (either JSON or Root format) using the `--geometry` option, together with the explicit `--mch-disable-geometry-from-ccdb`option.

To test it one can use e.g. a `sampler-transformer-sink` pipeline as such :

```
o2-mch-clusters-sampler-workflow
    -b --nEventsPerTF 1000 --infile someclusters.data |
o2-mch-clusters-transformer-workflow
    -b --mch-disable-geometry-from-ccdb --geometry Detectors/MUON/MCH/Geometry/Test/ideal-geometry-o2.json |
o2-mch-clusters-sink-workflow
    -b --txt --outfile global-clusters.txt --no-digits --global
```

