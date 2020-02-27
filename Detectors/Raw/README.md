<!-- doxy
\page refDetectorsRaw Raw Data 
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

## RawFileWriter

A class to facilitate writing to files the MC data converted to raw data payload. Payload will be formatted according to CRU data
specifications and should be simular to produced by the readout.exe (and can be pumped into the DPL workflows)

The detector code for MC to raw conversion should instatiate the RawFileWriter object and:

*   register all GBT links for which it wants to write data, providing either `cruID`, `linkID`, `CRU endPointID` and `feeID`
or an RDH containing similar infomation. It also can provide a file name to write the data of this link. Multiple links may
write their data to the same file:
```cpp
using namespace o2::raw;
RawFileWriter writer;
writer.registerLink(fee_0, cru_0, link_0, endpoint_0, "outfile_0.raw");
writer.registerLink(fee_1, cru_0, link_1, endpoint_0, "outfile_0.raw");
..
// or
o2::header::RawDataHeader rdh; // by default, v4 is used currently.
rdh.feeId = feeX;
rdh.cruID = cruY;
rdh.linkID = linkZ;
rdh.endPointID = endpointQ;
writer.registerLink( rdh, "outfile_f.raw");
```
*   add raw data payload to the `RawFileWriter`, providing the link information and the `o2::InteractionRecord` corresponding to payload.
The data must be provided as a `gsl::span<char>`` and contain only detector GBT (16 bytes words) payload. The annotation by RDH,
formatting to CRU pages and eventual splitting of the large payload to multiple pages will be done by the `RawFileWriter`.
```cpp
writer.addData(cru_0, link_0, endpoint_0, {bc, orbit}, gsl::span( (char*)payload_0, payload_0_size ) );
...
o2::InteractionRecord ir{bc, orbit};
writer.addData(rdh, ir, gsl::span( (char*)payload_f, payload_f_size ) );
```

The `RawFileWriter` will take care of writing created CRU data to file in `super-pages` whose size can be set using
`writer.setSuperPageSize(size_in_bytes)` (default in 1 MB).

The link buffers will be flushed and the files will be closed by the destor of the `RawFileWriter`, but this action can be
also triggered by `write.close()`.

In case detector link payload for given HBF exceeds the maximum CRU page size of 8KB (including the RDH added by the writer;
this may happen even if it the payload size is less than 8KB, since it might be added to already partially populated CRU page of
the same HBF) it will write on the page only part of the payload and carry over the rest on the extra page(s).
By default the RawFileWriter will simply chunk payload as it considers necessary, but some 
detectors want their CRU pages to be self-consistent: they add to their payload detector-specific trailer and header in the
beginning and in the end of the CRU page. In case part of the data is carried over to another page, this extra information
need to be repeated (possibly, with some modifications).
In orger to customize payload splitting, detector code may provide implement and provide to the `RawFileWriter` a
callback function with the signature:
```cpp
int carryOverMethod(const RDH& rdh, const gsl::span<char> data, const char* ptr, int maxSize, int splitID, std::vector<char>& trailer, std::vector<char>& header) const
```

The role of this method is to suggest to the writer how to split the payload: if it was provided to the RawFileWriter using
`writer.setCarryOverCallBack(pointer_on_the_converter_class);` then the `RawFileWriter` will call it before splitting.

It provides to the converter carryOverMethod method the following info:
```cpp
rdh     : RDH of the CRU page being written
data    : original payload received by the RawFileWriter
ptr     : pointer on the data in the payload which was not yet added to the link CRU pages
maxSize : maximum size (multiple of 16 bytes) of the bloc starting at ptr which it can 
          accomodate at the current CRU page (i.e. what it would write by default)
splitID : number of times this payload was already split, 0 at 1st call
trailer : what it wants to add in the end of the CRU page where the data starting from ptr
          will be added. The trailer is supplied as an empy vector, which carryOverMethod
          may populate, but its size must be multiple of 16 bytes.
header  : what it wants to add right after the RDH of the new CRU page before the rest of
          the payload (starting at ptr+actualSize) will be written
```

The method mast return actual size of the bloc which can be written (`<=maxSize`).
If this method populates the trailer, it must ensure that it returns the actual size such that
`actualSize + trailer.size() <= maxSize`.
In case returned `actualSize` is 0, current CRU page will be closed w/o adding anything, and new
query of this method will be done on the new CRU page.

Additionally, in case detector wants to add some information between `empty` HBF packet's opening and
closing RDHs (they will be added automatically using the HBFUtils functionality for all HBFs w/o data
between the orbit 0 and last orbit of the TF seen in the simulations), it may implement a callback method
```cpp
void emptyHBFMethod(const RDH& rdh, std::vector<char>& toAdd) const
```
If such method was registered using `writer.setEmptyPageCallBack(pointer_on_the_converter_class)`, then for every
empty HBF the writer will call it with
```cpp
rdh     : RDH of the CRU page opening empty RDH
toAdd   : a vector (supplied empty) to be filled to a size multipe of 16 bytes
```

The data `toAdd` will be inserted between the star/stop RDHs of the empty HBF.

For further details see  ``ITSMFT/common/simulation/MC2RawEncoder`` class and the macro
`Detectors/ITSMFT/ITS/macros/test/run_digi2rawVarPage_its.C` to steed the MC to raw data conversion.

## RawFileReader

A class for parsing raw data file(s) with "variable-size" CRU format.
For very encountered link the DPL `SubSpecification` is assigned according to the formula used by DataDistribution:
```cpp
SubSpec = (RDH.cruID<<16) | ((RDH.linkID + 1)<<(RDH.endPointID == 1 ? 8 : 0)); 
```
This `SubSpecification` is used to define the DPL InputSpecs (and match the OutputSpecs).
An `exception` will be thrown if 2 different link (i.e. with different RDH.feeId) of the same detector will be found to have the same SubSpec (i.e. the same cruID, linkID and PCIe EndPoint).
Also, an `exception` is thrown if the block expected to be a RDH fails to be recognized as RDH.

Since the raw data (RDH) does not contain any explicit information equivalent to `o2::header::DataOrigin` and `o2::header::DataDescription` needed to create a valid DPL `OutputSpec`,
for every raw data file provided to the `RawFileReader` the user can attach either explicitly needed origin and/or description, or to use default ones (which also can be modified by the user).
By default, `RawFileReader` assumes `o2::header::gDataOriginFLP` and `o2::header::gDataDescriptionRawData` (leading to `FLP/RAWDATA` OutputSpec).

The typical initialization of the `RawFileReader` is:
```cpp
using namespace o2::raw;
RawFileReader reader;
// optionaly override the default data origin by valid entry from `DataHeader.h`
reader.setDefaultDataOrigin("ITS");
// optionaly override the default data description by valid entry from `DataHeader.h`
reader.setDefaultDataDescription("RAWDATA");
// provide input file with explicit origin and description
reader.addFile("raw_data_file0.raw", o2::header::gDataOriginTPC, o2::header::gDataDescriptionRawData);
// provide raw dat file name only, default origin and description will be used
reader.addFile("raw_data_file1.raw");

// make reader to preprocess all input files, build the statistics for every GBT link encountered
and optionally check their conformity with expected CRU data format (see o2-raw-file-check below)
reader.init();
```

Note, that alternatively the input to the reader may be provided via text configuration file as:
```cpp
std::string inputConfig; // name should be assigned, empty string will be ignored
RawFileReader reader(inputConfig);
// if needed, add extra files as above
reader.init();
```

The input configuration must have following format (parsed using `Common-O2::ConfigFile`):
```cpp
# comments lines start with #

# optional, will override defaults set by RawFileReader
[defaults]
# optional, initial default is FLP 
dataOrigin = ITS
# optional, initial default is RAWDATA
dataDescription = RAWRDATA    

[input-0]
#optional, if missing then default is used
dataOrigin = ITS
#optional, if missing then default is used
dataDescription = RAWDATA     
filePath = path_and_name_of_the_data_file0

[input-1]
dataOrigin = TPC              
filePath = path_and_name_of_the_data_file1

[input-2]
filePath = path_and_name_of_the_data_file2

#...
# [input-XXX] blocks w/o filePath will be ignoder, XXX is irrelevant

```

The typical loop to read the data from already initialized reader is: 
```cpp
int nlinks = reader.getNLinks();
bool readPerTF = true; // data can be read per TF or per HBF

while(1) {
  int tfID = reader.getNextTFToRead();
  if (tfID >= reader.getNTimeFrames()) {
    LOG(INFO) << "nothing left to read after " << tfID << " TFs read";
    break;
  }
  std::vector<char> dataBuffer; // where to put extracted data  
  for (int il = 0; il < reader.getNLinks(); il++) {
    auto& link = reader.getLink(il);

    if (readPerTF) {       // read data per TF
      auto sz = link.getNextTFSize(); // size in bytes needed for the next TF of this link
      dataBuffer.resize(sz);
      link.readNextTF(dataTF.data());
      // use data ...
    }    
    else {                // read data per HBF 
      int nHBF = link.getNHBFinTF(); // number of HBFs in the TF
      for (int ihb=0;ihb<nHBF;ihb++) {
        auto sz = link.getNextHBSize(); // size in bytes needed for the next HBF of this link
	dataBuffer.resize(sz);
	link.readNextHBF(dataTF.data());
	// use data ...
      }
    }
  }
  reader.setNextTFToRead(++tfID);
}
```

Note: every input file may contain data from different CRUs or GBT links, but mixing of different detectors specifications (defined by
`o2::header::DataOrigin` and  `o2::header::DataDescription`) is not allowed.
Data from the same detector may be split to multiple files link-wise and/or time-wise. In the latter case the input files must be provided ordered in time (HBF orbit in the RDH).

## Raw data file reader workflow 

```cpp
o2-raw-file-reader-workflow
  ...
  --conf arg                            configuration file to init from (obligatory)
  --loop arg (=0)                       loop N times (infinite for N<0)
  --message-per-tf                      send TF of each link as a single FMQ 

```

The workflow takes an input from the configuration file (as described in `RawFileReader` section), reads the data and sends them as DPL messages
with the `OutputSpec`s indicated in the configuration file (or defaults). Each link data gets `SubSpecification` according to DataDistribution
scheme.

If `--loop` argument is provided, data will be re-played in loop.

At every invocation of the device `processing` callback a full TimeFrame for every link will be messaged as a multipart of N-HBFs messages (one for each HBF in the TF)
in a single `FairMQPart` per link (as the StfBuilder ships the data).
In case the `--message-per-tf` option is asked, the whole TF is sent as the only part of the `FairMQPart`.

The standard use case of this workflow is to provide the input for other worfklows using the piping, e.g.
```cpp
o2-raw-file-reader-workflow --conf myConf.cfg | o2-dpl-raw-parser
```

## Raw data file checker (standalone executable)

```cpp
Usage:   o2-raw-file-check [options] file0 [... fileN]
Options:
  -h [ --help ]                     print this help message.
  -c [ --conf ] arg                 read input from configuration file
  -v [ --verbosity ] arg (=0)    1: long report, 2 or 3: print or dump all RDH
  -s [ --spsize ]    arg (=1048576) nominal super-page size in bytes
  -t [ --hbfpertf ]  arg (=256)     nominal number of HBFs per TF
  --nocheck-packet-increment        ignore /Wrong RDH.packetCounter increment/
  --nocheck-page-increment          ignore /Wrong RDH.pageCnt increment/
  --nocheck-stop-on-page0           ignore /RDH.stop set of 1st HBF page/
  --nocheck-missing-stop            ignore /New HBF starts w/o closing old one/
  --nocheck-starts-with-tf          ignore /Data does not start with TF/HBF/
  --nocheck-hbf-per-tf              ignore /Number of HBFs per TF not as expected/
  --nocheck-tf-per-link             ignore /Number of TFs is less than expected/
  --nocheck-hbf-jump                ignore /Wrong HBF orbit increment/
  --nocheck-no-spage-for-tf         ignore /TF does not start by new superpage/
  (input files are optional if config file was provided)
```

Allows to check the correctness of CRU data (real or simulated) stored in the binary file.
Multiple files can be checked, with each file containing data for the same or distinct group of links.

Apart from the eventual `exception` produced for unrecognizable RDH or 2 links with the same cruID, linkID and PCIe EndPoint but different feeId (see `RawFileReader`),
the following errors (check can be disabled by corresponding option) are reported (as `ERROR`) for every GBT link while scanning each file
(the error counter of each link is incremented for any of this errors):

*   RawDataHeader packet counter (`RDH.packetCounter`) is not incremented by 1 (`--nocheck-packet-increment` to disable check)
*   RawDataHeader page counter (`RDH.pageCnt`) is not incremented by 1 (`--nocheck-page-increment` to disable)
*   `RDH.stop` is set at the very 1st page of the HBF (`--nocheck-stop-on-page0` to disable)
*   New HBF starts while the previous one was not yet closed, i.e. no page `RDH.stop = 1` received (`--nocheck-missing-stop` to disable)
*   Data of every link starts with TF (`--nocheck-starts-with-tf` to disable)
*   Number of HBFs in TF differs from the nominal /D=256, can be imposed by user/ (`--nocheck-hbf-per-tf` to disable)
*   The orbit/BC of the new HBF does not differ from previous one by 1 orbit exactly (`--nocheck-hbf-jump` to disable)
*   New TF starts not from the beginning of the CRU SuperPage; not reported for single-link files since in that case there is no way delimit a superpage (`--nocheck-no-spage-for-tf` to disable)

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
