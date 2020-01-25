<!-- doxy
\page DetectorsRaw Library
/doxy -->

# Raw data tools

## HBFUtil

Utility class for interaction record -> HBF conversion and sampling of IRs for which the HBF RDH should be added to the raw data from the CRU.
Should be used for MC->raw conversion to ensure that:

*   all detectors set the HBF and TF flags in a synchronized way

*   all TFs are complete (contain nominal /D=256/ number of HBFs with contiguously incremeneting orbit numbers):
for any given `payloadIR` (InteractionRecord = orbit/bc) for which payload is being created,
```cpp
using IR = o2::InteractionRecord;
HBFUtil::fillHBIRvector(std::vector<IR>& dst, const IR& fromIR, const IR& payLoadIR);
```
will fill the provided `dst` vector with the IR's (between `fromIR` and `payLoadIR`) for which the HBFs should be
generated and flushed to raw data **before** filling the payload of `payLoadIR`.

See `Detectors/Raw/test/testHBFUtils.cxx` for details of usage of this class (also check the jira ticket about the [MC->Raw conversion](https://alice.its.cern.ch/jira/browse/O2-972)).

## RawFileReader

A class for parsing raw data file(s) with "variable-size" CRU format.
For very encountered link the DPL `SubSpecification` is assigned according to the formula used by DataDistribution:
```cpp
SubSpec = (RDH.cruID<<16) | ((RDH.linkID + 1)<<(RDH.endPointID == 1 ? 8 : 0)); 
```
This `SubSpecification` is used to define the DPL InputSpecs (and match the OutputSpecs).
A `FATAL` will be produced if 2 different link (i.e. with different RDH.feeId) will be found to have the same SubSpec (i.e. the same cruID, linkID and PCIe EndPoint).
Also, a `FATAL` is produced if the block expected to be a RDH fails to be recognized as RDH.

TODO <em>At the moment only preprocissing part of input files (used in the o2-raw-filecheck descibed below) is operational.
The parsing to STF is in development</em>.

## Raw data file checker (standalone executable)

```cpp
Usage:   o2-raw-filecheck [options] file0 [... fileN]
Options:
  -h [ --help ]                     print this help message.
  -v [ --verbosity ] arg (=0)       1: print RDH on error, 2: print all RDH
  -s [ --spsize    ] arg (=1048576) nominal super-page size in bytes
  -t [ --hbfpertf  ] arg (=256)     nominal number of HBFs per TF
```

Allows to check the correctness of CRU data (real or simulated) stored in the binary file.
Multiple files can be checked, with each file containing data for the same or distinct group of links.

Apart from the eventual `FATAL` produced for unrecognizable RDH or 2 links with the same cruID, linkID and PCIe EndPoint but different feeId (see `RawFileReader`),
the following errors are reported (as `ERROR`) for every GBT link while scanning each file (the error counter of each link is incremented for any of this errors):

*   RawDataHeader packet counter (`RDH.packetCounter`) is not incremented by 1
*   RawDataHeader page counter (`RDH.pageCnt`) is not incremented by 1
*   New HBF starts with `RDH.pageCnt > 0`
*   `RDH.stop` is set at the very 1st page of the HBF
*   New HBF starts while the previous one was not yet closed (no page `RDH.stop = 1` received)
*   Number of HBFs in TF differs from the nominal (D=256, can be imposed by user)
*   The orbit/BC of the new HBF does not differ from previous one by 1 orbit exactly
*   New TF starts not from the beginning of the CRU SuperPage (not reported for single-link files since in that case there is no way delimit a superpage)

After scanning each file, for every link the largest SuperPage size (as the size of the largest contiguos block containing data of this link).
A warning is printed if this size exceeds the nominal SuperPage size (D=1MB, can be imposed by user).

For the correct data the output should look like this:
```cpp
o2-raw-filecheck ALPIDE_DATA/rawits_pseudoCalib_RU46_49.bin 
[INFO] RawDataHeader v4 is assumed
[INFO] #0: 182108928 bytes scanned, 69552 RDH read for 12 links from ALPIDE_DATA/rawits_pseudoCalib_RU46_49.bin

Summary of preprocessing:
L0   | Spec:0x3014657  FEE:0x2012 CRU:  46 Lnk:  0 EP:0 | SPages:   8 Pages:  4296 TFs:     8 with   2048 HBF in    8 blocks (0 err)
L1   | Spec:0x3014658  FEE:0x2112 CRU:  46 Lnk:  1 EP:0 | SPages:   8 Pages:  4296 TFs:     8 with   2048 HBF in    8 blocks (0 err)
L2   | Spec:0x3014659  FEE:0x2212 CRU:  46 Lnk:  2 EP:0 | SPages:   8 Pages:  4296 TFs:     8 with   2048 HBF in    8 blocks (0 err)
L3   | Spec:0x3080193  FEE:0x2013 CRU:  47 Lnk:  0 EP:0 | SPages:   8 Pages:  4296 TFs:     8 with   2048 HBF in    8 blocks (0 err)
L4   | Spec:0x3080194  FEE:0x2113 CRU:  47 Lnk:  1 EP:0 | SPages:   8 Pages:  4296 TFs:     8 with   2048 HBF in    8 blocks (0 err)
L5   | Spec:0x3080195  FEE:0x2213 CRU:  47 Lnk:  2 EP:0 | SPages:   8 Pages:  4296 TFs:     8 with   2048 HBF in    8 blocks (0 err)
L6   | Spec:0x3145729  FEE:0x3000 CRU:  48 Lnk:  0 EP:0 | SPages:  31 Pages:  7096 TFs:     8 with   2048 HBF in   31 blocks (0 err)
L7   | Spec:0x3145730  FEE:0x3100 CRU:  48 Lnk:  1 EP:0 | SPages:  31 Pages:  7096 TFs:     8 with   2048 HBF in   31 blocks (0 err)
L8   | Spec:0x3145731  FEE:0x3200 CRU:  48 Lnk:  2 EP:0 | SPages:  31 Pages:  7696 TFs:     8 with   2048 HBF in   31 blocks (0 err)
L9   | Spec:0x3211265  FEE:0x3001 CRU:  49 Lnk:  0 EP:0 | SPages:  31 Pages:  7096 TFs:     8 with   2048 HBF in   31 blocks (0 err)
L10  | Spec:0x3211266  FEE:0x3101 CRU:  49 Lnk:  1 EP:0 | SPages:  31 Pages:  7096 TFs:     8 with   2048 HBF in   31 blocks (0 err)
L11  | Spec:0x3211267  FEE:0x3201 CRU:  49 Lnk:  2 EP:0 | SPages:  31 Pages:  7696 TFs:     8 with   2048 HBF in   31 blocks (0 err)
Largest super-page: 1047008 B, largest TF: 4070048 B
Real time 0:00:00, CP time 0.120
```
