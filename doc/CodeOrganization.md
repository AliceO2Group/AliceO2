Code organisation
=================

## Overview

The AliceO2 repository is subdivided in a number of sub-modules.
The _per detector_ sub-modules are grouped under the Detectors directory.
The _per function_ are in the top directory or grouped, e.g. Utilites.

A typical submodule looks like :
~~~~
.
|-- Common
|   |-- CMakeLists.txt
|   |-- doc
|   |   `-- All documentation files
|   |-- include
|   |   `-- Common
|   |       `-- Factory.h
|   |-- src
|   |   |-- Internal.h
|   |   `-- Factory.cxx
|   `-- test
|       `-- TestFactory.cxx

~~~~

Depending on the case, some subdirectories can be voluntarily left out or added.
The headers go to the include directory if they are part of the interface, in the src otherwise.

### Other repositories in AliceO2Group

Other repositories in the AliceO2Group follow the same structure.

## Principles

A number of principles were agreed on that resulted in the above code organisation :

* A _module_ is a set of code closely related sharing an interface that can result in one or more libraries.
* Favour is given to extracting large common components(modules/projects) into their own repositories within
  AliceO2Group in github.
* AliceO2 therefore becomes a thinner repo containing :
  * Detector specific code (e.g. related to reconstruction, simulation, calibration or qc).
  * Commonalities (e.g. DataFormat, Steer-like), i.e. things other components depend on and that have not been extracted to their own repo.
  * Global algorithms (e.g. global tracking), i.e. things that depend on several detectors.
* The directory structure can be either per detector or per function or a mixture.
  The AliceO2 repository has a mixture of _per detector_ and _per function_ sub-modules with corresponding sub-structure.
* Dependencies are defined centrally as _buckets_.
* Each sub-module generates a single library linked against the dependencies defined in a single bucket.
* sub-modules' executable(s) link against the same bucket as the library and the library itself.
* Horizontal dependencies are in general forbidden (between sub-modules at the same level) (?)
* Naming : camel-case
  * What is repeated / structural starts with a lower case letter (e.g. src, include, test).
  * The rest (labels, unique names) start with an upper case letter (e.g. Common, Detectors).
* Why are headers in `MyModule/include/MyModule` and not directly in `MyModule/include ?`
  * The difficulty here is that we have a number of constraints. First the headers must be installed in a directory
 named after the module. Second the code which uses the headers must include `MyModule/xyz.h` and it must work
 whether it is inside AliceO2 or in a different repo, i.e. whether the headers are installed or they are internal.
 When evaluating the different options we ended up with this not-totally-perfect solution because all other solutions
 broke one of the constraints or required a massive hurdle of CMake magic. If someone comes up with a different working
 solution we would happily consider it.