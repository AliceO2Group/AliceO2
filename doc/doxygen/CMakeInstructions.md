Code organisation and build
=

## Principles
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

## Repository organisation
TODO

## CMake instructions

* CMakeLists.txt organisation
* Macros available to the users
* Their parameters, what they are, whether they are required or not and in which context

## Examples

The two modules ExampleModule1 and ExampleModule2 show a basic implementation of modules following
the guidelines described above. 