# FairRootBase

This directory holds the part of [FairRoot](https://github.com/FairRootGroup/FairRoot) that the O2
simulation uses: the VMC application, module and detector base classes, the run, sink and root
manager, the generic stack, the event headers and a few helpers around them. The code was taken from
FairRoot `v18.4.9-alice3`, commit `a16083b13989d08de596b4fedbde4df2c3880a35`, which is the version
ALICE has been building against.

The files keep their original copyright and licence headers. FairRoot is distributed under the GNU
Lesser General Public Licence version 3. The licence text is in [LICENSE](LICENSE) and the copyright
statement in [COPYRIGHT](COPYRIGHT), with [AUTHORS](AUTHORS) and [CONTRIBUTORS](CONTRIBUTORS), all copied verbatim
from FairRoot. Each modified file carries a note below its licence header.

Changes with respect to the upstream sources, made in 2026:

- Code paths that O2 does not use were removed. This covers the runtime parameter database and the
  geometry parameter containers, among others. The affected files are `FairMCApplication`,
  `FairModule`, `FairRun`, `FairRunSim`, `FairSink` and `FairField`.
- `FairModule` stops with a clear message if its geometry import is used, since O2 does not set it up.
- `FairRootBaseLinkDef.h` is new and builds one dictionary for the directory.

The sources keep their upstream formatting and are excluded from clang-format.

We are grateful to the FairRoot developers for this code and for many years of collaboration.
