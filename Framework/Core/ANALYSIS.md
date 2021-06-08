<!-- doxy
\page refFrameworkCoreANALYSIS Core ANALYSIS
/doxy -->

The comprehensive documentation of the O2 Analysis Framework is located at [https://aliceo2group.github.io/analysis-framework/](https://aliceo2group.github.io/analysis-framework/).

```note
Please do not update this document, but contribute directly to the O2 Analysis Framework documentation at  

https://github.com/AliceO2Group/analysis-framework/tree/master/docs/framework

status May 17, 2021
```

---
---



##  Core ANALYSIS

This document is WIP and provides an idea of what kind of API to expect from the DPL enabled analysis framework. APIs are neither final nor fully implemented in O2.

# Analysis Task infrastructure on top of DPL

In order to simplify analysis we have introduced an extension to DPL which allows to describe an Analysis in the form of a collection of AnalysisTask.

In order to create its own task, as user needs to create your own Task deriving from AnalysisTask.

```cpp
struct MyTask : AnalysisTask {
};
```

such a task can then be added to a workflow via the `adaptAnalysisTask` helper. A full blown example can be built with:

```cpp
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

struct MyTask : AnalysisTask {
};

defineDataProcessing() {
  return {
    adaptAnalysisTask<MyTask>(TaskName{"my-task-unique-name"});
  };
}
```

> **Implementation details**: `AnalysisTask` is simply a `struct`. Since `struct` default inheritance policy is `public`, we can omit specifying it when declaring MyTask.
>
> `AnalysisTask` will not actually provide any virtual method, as the `adaptAnalysis` helper relies on template argument matching to discover the properties of the task. It will come clear in the next paragraph how this allow is used to avoid the proliferation of data subscription methods.   

## Processing data

### Simple subscriptions

Once you have an `AnalysisTask` derived type, the most generic way which you can use to process data is to provide a `process` method for it.

Depending on the arguments of such a function, you get to iterate on different parts of the AOD content.

For example:

```cpp
struct MyTask : AnalysisTask {
  void process(o2::aod::Tracks const& tracks) {
    ...
  }
};
```

will allow you to get a per time frame collection of tracks. You can then iterate on the tracks using the syntax:

```cpp
for (auto &track : tracks) {
  tracks.alpha();
}
```

Alternatively you can subscribe to tracks one by one via (notice the missing `s`):

```cpp
struct MyTask : AnalysisTask {
  void process(o2::aod::Track const& track) {
    ...
  }
};
```

This has the advantage that you might be able to benefit from vectorization / parallelization.

> **Implementation notes**: as mentioned before, the arguments of the process method are inspected using template argument matching. This way the system knows at compile time what data types are requested by a given `process` method and can create the relevant DPL data descriptions. 
>
> The distinction between `Tracks` and `Track` above is simply that one refers to the whole collection, while the second is an alias to `Tracks::iterator`.  Notice that we assume that each collection is of type `o2::soa::Table` which carries meta data about the dataOrigin and dataDescription to be used by DPL to subscribe to the associated data stream.

### Navigating data associations

For performance reasons, data is organized in a set of flat table and navigation between objects of different tables has to be expressed explicitly in the `process` method. So if you want to get all the tracks for a specific collision, you will have to implement:

```cpp
void process(o2::aod::Collision const& collision, o2::aod::Tracks &tracks) {
...
}
```

the above will be called once per collision found in the time frame, and `tracks` will allow you to iterate on all the tracks associated to the given collision.

Alternatively, you might not require to have all the tracks at once and you could do with:

```cpp
void process(o2::aod::Collection const& collision, o2::aod::Track const& track) {
}
```

also in this case the advantage is that your code might be up for parallelization and vectorization.

Notice that you are not limited to two different collections, but you could specify more. E.g.: 

```cpp
void process(o2::aod::Collection const& collision, o2::aod::V0 const& v0, o2::aod::Tracks const& tracks) {
}
```

will be invoked for each v0 associated to a given collision and you will be given the tracks associated to it.

This means that each subsequent argument is associated to all the one preceding it.

### Processing related tables

For performance reasons, sometimes it's a good idea to split data in separate tables, so that once can request only the subset which is required for a given task. For example, so far the track related information is split in three tables: `Tracks`, `TrackCovs`, `TrackExtras`.

However you might need to get all the information at once. This can be done by asking for a `Join` table in the process method:

```cpp
struct MyTask : AnalysisTask {

  void process(Join<Tracks, TracksExtras> const& mytracks) {
    for (auto& track : mytracks) {
      if (track.length()) {  // from TrackExtras
        tracks.alpha();      // from Tracks
      }
    }
  }
}
```

## Creating new collections

In order to create new collections of objects, you need two things. First of all you need to define a datatype for it, then you need to specify that your analysis task will create such an object. Notice that in a given workflow, only one task is allowed to create a given type of object.

### Introducing a new data type

In order to define the datatype you need to use `DEFINE_SOA_COLUMN` and `DEFINE_SOA_TABLE` helpers, defined in `ASoA.h`. Assuming you want to extend the standard AOD format you will also need `Framework/AnalysisDataModel.h`. For example, to define an extra table where to define phi and eta, you first need to define the two columns:

```cpp
#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"

namespace o2::aod {

namespace etaphi {
DECLARE_SOA_COLUMN(Eta, eta, float, "fEta");
DECLARE_SOA_COLUMN(Phi, phi, float, "fPhi");
}
}
```

and then you put them together in a table:

```cpp
namespace o2::aod {
DECLARE_SOA_TABLE(EtaPhi, "AOD", "ETAPHI",
                  etaphi::Eta, etaphi::Phi);
}
```

Notice that tables are actually just a collections of columns.

### Creating objects for a new data type

Once you have the new data type defined, you can have a task producing it, by using the `Produces` helper:

```cpp
struct MyTask : AnalysisTask {
  Produces<o2::aod::EtaPhi> etaphi;

  void process(o2::aod::Track const& track) {
    etaphi(calculateEta(track), calculatePhi(track));
  }
};
```

the `etaphi` object is a functor that will effectively act as a cursor which allows to populate the `EtaPhi` table. Each invocation of the functor will create a new row in the table, using the arguments as contents of the given column. By default the arguments must be given in order, but one can give them in any order by using the correct column type. E.g. in the example above:

```cpp
etaphi(track::Phi(calculatePhi(track), track::Eta(calculateEta(track)));
```

### Adding dynamic columns to a data type

Sometimes columns are not backed by actual persisted data, but they are merely
derived from it. For example you might want to have different representations
(e.g. spherical, cylindrical) for a given persistent representation. You can
do that by using the `DECLARE_SOA_DYNAMIC_COLUMN` macro.

```cpp
namespace point {
DECLARE_SOA_COLUMN(X, x, float, "fX");
DECLARE_SOA_COLUMN(Y, y, float, "fY");
}

DECLARE_SOA_DYNAMIC_COLUMN(R2, r2, [](float x, float y) { return x*x + y+y; });

DECLARE_SOA_TABLE(Point, "MISC", "POINT", X, Y, (R2<X,Y>));
```

Notice how the dynamic column is defined as a stand alone column and binds to X and Y
only when you attach it as part of a table.

### Executing a finalization method, post run

Sometimes it's handy to perform an action when all the data has been processed, for example executing a fit on a histogram we filled during the processing. This can be done by implementing the postRun method.

### Creating histograms

New tables are not the only kind on objects you want to create, but most likely you would like to fill histograms associated to the objects you have calculated.

You can do so by using the `Histogram` helper:

```cpp
struct MyTask : AnalysisTask {
  Histogram etaHisto;

  void process(o2::aod::EtaPhi const& etaphi) {
    etaHisto.fill(etaphi.eta());
  }
};
```

# Creating new columns in a declarative way

Besides the `Produces` helper, which allows you to create a new table which can be reused by others, there is another way to define a single column,  via the `Defines` helper.

```cpp
struct MyTask : AnalysisTask {
  Defines<track::Eta> eta = track::alpha;
};
```

## Filtering and partitioning data

Given a process function, one can of course define a filter using an if condition:

```cpp
struct MyTask : AnalysisTask {
  void process(o2::aod::EtaPhi const& etaphi) {
    if (etaphi.phi() > 1 && etaphi.phi < 1) {
      ...
    }
  }
};
```

however this has the disadvantage that the filtering will be done for every
task which has similar or more restrictive conditions. By declaring your
filters upfront you can not only simplify your code, but allow the framework to
optimize your processing.  To do so, we provide two helpers: `Filter` and
`Partition`. 

### Upfront filtering

The most common kind of filtering is when you process objects only if one of its
properties passes a certain criteria. This can be specified with the `Filter` helper.

```cpp
struct MyTask : AnalysisTask {
  Filter<Tracks> ptFilter = track::pt > 1;

  void process(Tracks const &filteredTracks) {
    for (auto& track : filteredTracks) {
    }
  }
};
```

filteredTracks will contain only the tracks in the table which pass the condition `track::pt > 1`. 

You can specify multiple filters which will be applied in a sequence effectively resulting in the intersection of all them.

You can also specify filters on associated quantities:

```cpp
struct MyTask : AnalysisTask {
  Filter<Collisions> collisionFilter = max(track::pt) > 1;

  void process(Collsions const &filteredCollisions) {
    for (auto& collision: collisions) {
    ...
    }
  }
};
```

will process all the collisions which have at least one track with `pt > 1`.

### Partitioning your inputs

Filtering is not the only kind of conditional processing one wants to do. Sometimes you need to divide your data in two or more partitions. This is done via the `Partition` helper:

```cpp
using namespace o2::aod;

struct MyTask : AnalysisTask {
  Partition<Tracks> leftTracks = track::eta < 0;
  Partition<Tracks> rightTracks = track::eta >= 0;

  void process(Tracks const &tracks) {
    for (auto& left : leftTracks(tracks)) {
      for (auto& right : rightTracks(tracks)) {
        ...
      }
    }
  }
};
```

i.e. `Filter` is applied to the objects before passing them to the `process` method, while `Select` objects can be used to do further reduction inside the `process` method itself. 

### Filtering and partitioning together

Of course it should be possible to filter and partition data in the same task. The way this works is that multiple `Filter`s are logically ANDed together and then they will get anded with the OR of all the `Select` specified selections.

### Configuring filters

One of the features of the current framework is the ability to customize on the fly cuts and selection. The idea is to allow that by having a `configurable("mnemonic-name-of-the-parameter")` helper which can be used to refer to configurable options. The previous example will then become:

```cpp
struct MyTask : AnalysisTask {
  Filter<Collisions> collisionFilter = max(track::pt) > configurable<float>("my-pt-cut");

  void process(Collsions const &filteredCollisions) {
    for (auto& collision: collisions) {
    ...
    }
  }
};
```

### Getting combinations (pairs, triplets, ...)
To get combinations of distinct tracks, helper functions from `ASoAHelpers.h` can be used. Presently, there are 3 combinations policies available: strictly upper, upper and full. `CombinationsStrictlyUpperPolicy` is applied by default if all tables are of the same type, otherwise `FullIndexPolicy` is applied.

```cpp
combinations(tracks, tracks); // equivalent to combinations(CombinationsStrictlyUpperIndexPolicy(tracks, tracks));
combinations(filter, tracks, covs); // equivalent to combinations(CombinationsUpperIndexPolicy(tracks, covs), filter, tracks, covs);
```

The number of elements in a combination is deduced from the number of arguments passed to `combinations()` call. For example, to get pairs of tracks from the same source, one must specify `tracks` table twice:

```cpp
struct MyTask : AnalysisTask {

  void process(Tracks const& tracks) {
    for (auto& [t0, t1] : combinations(CombinationsStrictlyUpperIndexPolicy(tracks, tracks))) {
      float pt = t0.pt();
      ...
    }
  }
};
```

The combination can consist of elements from different tables (of different kinds):

```cpp
struct MyTask : AnalysisTask {

  void process(Tracks const& tracks, TracksCov const& covs) {
    for (auto& [t0, c1] : combinations(CombinationsFullIndexPolicy(tracks, covs))) {
      ...
    }
  }
};
```

One can get combinations of elements with the same value in a given column. Input tables do not need to be the same but each table must contain the column used for categorizing. Additionally, you can specify a value to be skipped for grouping as well as the number of elements to be matched with first element in a combination. Again, full, strictly upper and upper policies are available:

```cpp
for (auto& [c0, c1] : combinations(CombinationsBlockStrictlyUpperIndexPolicy("fRunNumber", 3, -1, collisions, collisions))) {
  // Pairs of collisions with same fRunNumber, max 3 pairs for each element in a given "fRunNumber" bin. Entries with fRunNumber == -1 are skipped.
}
for (auto& [c0, t1] : combinations(CombinationsBlockFullIndexPolicy("fX", 200, -1, collisions, tracks)));
```

For better performance, if the same table is used, `Block{Full,StrictlyUpper,Upper}SameIndex` policies should be preferred. `selfCombinations()` are a shortuct to apply StrictlyUpperSameIndex policy:

```cpp
for (auto& [c0, c1] : combinations(CombinationsBlockFullSameIndexPolicy("fRunNumber", 3, -1, collisions, collisions)));
for (auto& [c0, c1] : selfCombinations("fRunNumber", 3, -1, collisions, collisions)) {
  // same as: combinations(CombinationsBlockStrictlyUpperSameIndexPolicy("fRunNumber", 3, -1, collisions, collisions));
}
```

It will be possible to specify a filter for a combination as a whole, and only matching combinations will be then output. Currently, the filter is applied to each element separately. Note that for filter version the input tables are mentioned twice, both in policy constructor and in `combinations()` call itself.

```cpp
struct MyTask : AnalysisTask {

  void process(Tracks const& tracks1, Tracks const& tracks2) {
    Filter triplesFilter = track::eta < 0;
    for (auto& [t0, t1, t2] : combinations(CombinationsFullIndexPolicy(tracks1, tracks2, tracks2), triplesFilter, tracks1, tracks2, tracks2)) {
      // Triples of tracks, each of them with eta < 0
      ...
    }
  }
};
```

### Saving tables to file

Produced tables can be saved to file as TTrees. This process is customized by various command line options of the internal-dpl-aod-writer. The options allow to specify which columns of which table are saved to which tree in which file.

**Please be aware, that the functionality of these options is preliminary and might be changed in future.**

The options to consider are:

* --aod-writer-keep
* --aod-writer-resfile
* --aod-writer-ntfmerge
* --aod-writer-json


#### --aod-writer-keep

`aod-writer-keep` is a comma-separated list of `DataOuputDescriptors`.

`aod-writer-keep`
```csh
DataOuputDescriptor1,DataOuputDescriptor2, ...
```

Each `DataOuputDescriptor` is a colon-separated list of 4 items

`DataOuputDescriptor`
```csh
table:tree:columns:file
```
and instructs the internal-dpl-aod-writer, to save the columns `columns` of table `table` as TTree `tree` in folder `TF_x` of file `file.root`. The selected columns are saved as separate TBranches of TTree `tree`.

By default `x` is incremented with every time frame. This behavior can be modified with the command line option `--ntfmerge`. The value of `aod-writer-ntfmerge` specifies the number of time frames to merge into one `TF_x` folder. 

The first item of a `DataOuputDescriptor` (`table`) is mandatory and needs to be specified, otherwise the `DataOuputDescriptor` is ignored. The other three items are optional and are filled by default values if missing.

The format of `table` is

`table`
```csh
AOD/tablename/0
```
`tablename` is the name of the table as defined in the workflow definition.

The format of `tree` is a simple string which names the TTree the table is saved to. If `tree` is not specified then `O2tablename` is used as TTree name.

`columns` is a slash(/)-separated list of column names., e.g.

`columns`
```csh
col1/col2/col3
```
The column names are expected to match column names of table `tablename` as defined in the respective workflow. Non-matching columns are ignored. The selected table columns are saved as separate TBranches with the same names as the corresponding table columns. If `columns` is not specified then all table columns are saved.

`file` finally specifies the base name of the file the tables are saved to. The actual file name is `file`.root. If `file` is not specified the default file name is used. The default file name can be set with the command line option `--aod-writer-resfile`. However, if `aod-writer-resfile` is missing then the default file name is set to `AnalysisResults_trees`.

##### Dangling outputs
The `aod-writer-keep` option also accepts the string "dangling" (or any leading sub-string of it). In
this case all dangling output tables are saved. For the parameters `tree`, `columns`, and
`file` the default values ([see table below](#priorities)) are used.

#### --aod-writer-ntfmerge

`aod-writer-ntfmerge` specifies the number of time frames which are merged into a given folder `TF_x`. By default this value is set to 1. `x` is incremented by 1 at every `aod-writer-ntfmerge` time frame.

#### --aod-writer-resfile

`aod-writer-resfile` specifies the default base name of the results files to which tables are saved. If in any of the `DataOutputDescriptors` the `file` value is missing it will be set to this default value.

#### --aod-writer-json

`aod-writer-json` specifies the name of a json-file which contains the full information needed to customize the behavior of the internal-dpl-aod-writer. It can replace the other three options completely. Nevertheless, currently all options are supported ([see also discussion below](#redundancy)).

An example file is shown in the highlighted field below. The relevant
information is contained in a json object `OutputDirector`. The
`OutputDirector` can include three different items:

  1. `resfile` is a string and corresponds to the `aod-writer-resfile` command line option  
  2.`aod-writer-ntfmerge` is an integer and corresponds to the `aod-writer-ntfmerge` command line option  
  3.`OutputDescriptors` is an array of objects and corresponds to the `aod-writer-keep` command line option. The objects are equivalent to the `DataOuputDescriptors` of the `aod-writer-keep` option and are composed of 4 items which correspond to the 4 items of a `DataOuputDescriptor`.
  
     a. `table` is a string  
     b. `treename` is a string  
     c. `columns` is an array of strings  
     d. `filename` is a string  
  
  
`Example json file for the internal-dpl-aod-writer`
```csh
{
  "OutputDirector": {
      "resfile": "defresults",
      "ntfmerge": 10,
      "OutputDescriptors": [
          {
            "table": "AOD/UNO/0",
            "columns": [
              "col1",
              "col2"
            ],
            "treename": "uno",
            "filename": "unoresults"
          },
          {
            "table": "AOD/DUE/0",
            "columns": [
              "col3"
            ],
            "treename": "due",
            "filename": "dueresults"
          }
      ]
  }
}
```
<a name="redundancy"></a>
The information provided with the json file and the information which can be provided with
the other command line options is obviously redundant. Anyway, currently all options can
be used together. Practically the json-file - if provided - is read first. Then parameters are reset with values specified by other command line options. If any parameter value is still unset then its default value is used.

This hierarchy of the options is summarized in the following table. The columns represent the command line options and the rows the parameters which can be set. The table elements specify the priority a given command line option has to set the value of a given parameter. The last column in the table is the default, which always has the lowest priority. The actual default value is the value shown between brackets.

<a name="priorities"></a>

| parameter\option | aod-writer-keep | aod-writer-resfile | aod-writer-ntfmerge | aod-writer-json | default |
|--------------|:----:|:--------:|:--------:|----------:|:-------:|
| `default file name` | -        | 1.       | -         | 2.      | 3. (AnalysisResults_trees)|
| `ntfmerge`   | -    | -        | 1.       | 2.        | 3. (1)  |
| `tablename`  | 1.   | -        | -        | 2.        | -       |
| `tree`       | 1.   | -        | -        | 2.        | 3. (`tablename`) |
| `columns`    | 1.   | -        | -        | 2.        | 3. (all columns)     |
| `file`       | 1.   | 2.       | -        | 3.        | 4. (`default file name`)|


#### Valid example command line options

```csh
--aod-writer-keep AOD/UNO/0
 # save all columns of table 'UNO' to TTree 'O2uno' in file `AnalysisResults_trees.root`
  
--aod-writer-keep AOD/UNO/0::c2/c4:unoresults
 # save columns 'c2' and 'c4' of table 'UNO' to TTree 'O2uno' in file 'unoresults.root`

--aod-writer-resfile myskim --aod-writer-ntfmerge 50 --aod-writer-keep AOD/UNO/0:trsel1:c1/c2,AOD/DUE/0:trsel2:c6/c7/c8
 # save columns 'c1' and 'c2' of table 'UNO' to TTree 'trsel1' in file 'myskim.root` and
 # save columns 'c6', 'c7' and 'c8' of table 'DUE' to TTree 'trsel2' in file 'myskim.root`.
 # Merge 50 time frames in each folder `TF_x`.

--aod-writer-json myconfig.json
 # according to the contents of myconfig.json
```

#### Limitations

If in any case two `DataOuputDescriptors` are provided which have equal combinations of
the `tree` and `file` parameters then the processing is stopped! It is not possible to save
two trees with equal name to a given file.


### Reading tables from files

The internal-dpl-aod-reader reads trees from root files and provides them as arrow tables to the requesting workflows. Its behavior is customized with the following command line options:

* --aod-file
* --aod-reader-json

#### --aod-file

`aod-file` takes a string as option value, which either is the name of the input root file or, if starting with an `@`-character, is an ASCII-file which contains a list of input files. 

```csh
--aod-file AnalysisResults_0.root
 # uses AnalysisResults_0.root as input file

--aod-file @AnalysisResults.txt
 # uses files listed in AnalysisResults.txt as input files

```

#### --aod-reader-json

'aod-reader-json' is a string and specifies a json file, which contains the
customization information for the internal-dpl-aod-reader. An example file is
shown in the highlighted field below. The relevant information is contained in
a json object `InputDirector`. The `InputDirector` can include the following
three items:

  1. `resfiles` is a string or an array of strings and corresponds to the `aod-file` command line option. As the `aod-file` option it can specify a single input file or, when the option value starts with a `@`-character, an ASCII file with a list of input files. In addition `resfiles` can be an array of strings, which contains a list of input files.
  2.`fileregex` is a regex string which is used to select the input files from the file list specified by `resfiles`.
  3.`InputDescriptors` is an array of objects, the so called `DataInputDescriptors`, which are composed of 4 items.
  
     a. `table` is a string and specifies the table to fill. The `table` needs to be provided in the format `AOD/tablename/0`, where `tablename` is the name of the table as defined in the workflow definition.  
     b. `treename` is a string and specifies the tree which is to be used to fill `table`  
     c. `resfiles` is either a string or an array of strings. It specifies a list of possible input files (see discussion of `resfiles` above).  
     d. `fileregex` is a regular expression string which is used to select the input files from the file list specified by `resfiles`  

The information contained in a `DataInputDescriptor` instructs the internal-dpl-aod-reader to fill table `table` with the values from the tree `treename` in folders `TF_x` of the files which are defined by `resfiles` and which names match the regex `fileregex`.

Of the four items of a `DataInputDescriptor`, `table` is the only required information. If one of the other items is missing its value will be set as follows:

  1. `treename` is set to `O2tablename` of the respective `table` item.  
  2. `resfiles` is set to `resfiles` of the `InputDirector` (1. item of the `InputDirector`). If that is missing, then the value of the `aod-file` option is used. If that is missing, then `AnalysisResults_trees.root` is used.  
  3. `fileregex` is set to `fileregex` of the `InputDirector` (2. item of the `InputDirector`). If that is missing, then `.*` is used.


`Example json file for the internal-dpl-aod-reader`
```csh
{
  "InputDirector": {
    "resfiles": "@resfiles.txt",
    "fileregex": ".*",
    "InputDescriptors": [
      {
        "table": "AOD/COLLISION/0",
        "treename": "uno",
        "resfiles": [
          "unoresults_1.root",
          "unoresults_2.root",
          "unoresults_3.root",
          "unoresults_4.root"
        ]
      },
      {
        "table": "AOD/DUE/0",
        "treename": "due",
        "resfiles": "@dueresults.txt",
        "fileregex": "(dueresults)(.*)"
      },
      {
        "table": "AOD/TRE/0",
        "treename": "tre"
      }
    ]
  }
```

When the internal-dpl-aod-reader receives the request to fill a given table `tablename` it searches in the provided `InputDirector` for the corresponding `InputDescriptor` and proceeds as defined there. However, if there is no corresponding `InputDescriptor` it falls back to the information provided by the `resfiles` and `fileregex` options of the `InputDirector` and uses `O2tablename` as `treename`.

#### Some practical comments

The `aod-reader-json` option allows to setup the reading of tables in a rather
flexible way. Here a few presumably practical cases are discussed:

  1. Let's assume a case where data from tables `tableA` and `tableB` need to
be processed together. Table `tableA` was previously saved as tree `tableA` to
files `tableAResults_n.root`, where `n` is a number and `tableB` was saved as
tree `tableB` to `tableBResults_n.root`. The following json-file could be used
to read these tables:

```csh
{
  # file resfiles.txt lists all tableAResults_x.root and tableBResults_x.root files.

  "InputDirector": {
    "resfiles": "@resfiles.txt",
    "InputDescriptors": [
      {
        "table": "AOD/tableA/0",
        "fileregex": "(tableAResult)(.*)"
      },
      {
        "table": "AOD/tableB/0",
        "fileregex": "(tableBResult)(.*)"
      }
    ]
  }
```

  2. In this case several tables need to be provided. All tables can be read from files `tableResult_n.root`, except for one table, namely `tableA`, which is saved as tree `treeA` in files `tableAResult_n.root`.
  
```csh
  # file resfiles.txt lists all tableResults_n.root and tableAResults_n.root files.

  "InputDirector": {
    "resfiles": "@resfiles.txt",
    "fileregex": "(tableResult)(.*)"
    "InputDescriptors": [
      {
        "table": "AOD/tableA/0",
        "treename": "treeA",
        "fileregex": "(tableAResult)(.*)"
      }
    ]
  }
```

#### Limitations

  1. It is required that all `InputDescriptors` have the same number of selected input files. This is internally checked and the processing is stopped if it turns out that this is not the case.
  2. The internal-dpl-aod-reader loops over the selected input files in the order as they are listed. It is the duty of the user to make sure that the order is correct and that the order in the file lists
of the various `InputDescriptors` are corresponding to each other.
  3. The regular expression `fileregex` is evaluated with the c++ Regular expressions library. Thus check there for the proper syntax of regexes.
  

### Possible ideas

We could add a template `<typename C...> reshuffle()` method to the Table class which allows you to reduce the number of columns or attach new dynamic columns. A template wrapper could
even be used to specify if a given dynamic column should be precalculated (or not). This would come handy to optimize the creation of a RowView, which could bind only the required (dynamic) columns. E.g.:

```cpp
for (auto twoD : points.reshuffle<point::X, point::Y, Cached<point::R>>()) {
...
} 
```
