<!-- doxy
\page refDetectorsRaw Raw Data
/doxy -->

# Raw data tools

## RDHUtils

This is a static utilities class with the setters and getters of the various fields of the `o2::headers::RAWDataHeaderVXXX` classes of different versions. They allow uniform set/get access either via RDH reference of its pointer irrespective of the header version (starting from V3).
Additionally, it uses a generic `o2::header::RDHAny` which can be morphed to any real RDH version (and vice versa).
The sample of the code which can access the RDH in a version-idependent way is:
```cpp
using namespace o2::raw;

void process(InputRecord& inputs) {
  // ...
  DPLRawParser parser(inputs);
  for (auto it = parser.begin(); it != parser.end(); ++it) {
    const auto& rdh = *reinterpret_cast<const o2::header::RDHAny*>(it.raw());
    RDHUtils::printRDH(rdh);
    auto feeID = RDHUtils::getFEEID(rdh);
    // ...
  }
  // ...
}
```

## HBFUtils

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

Extra data member `HBFUtil.maxNOrbits` (= highest possible orbit by default) is used only in the RawFileWriter as a maximum orbit number (with respect to 1st orbit) for which detector payload is considered.

See `Detectors/Raw/test/testHBFUtils.cxx` for details of usage of this class (also check the jira ticket about the [MC->Raw conversion](https://alice.its.cern.ch/jira/browse/O2-972)).

## RawFileWriter

A class to facilitate writing to files the MC data converted to raw data payload. Payload will be formatted according to CRU data
specifications and should be simular to produced by the o2-readout-exe (and can be pumped into the DPL workflows)

The detector code for MC to raw conversion should instatiate the RawFileWriter object and:

*   register all GBT links for which it wants to write data, providing either `cruID`, `linkID`, `CRU endPointID` and `feeID`
or an RDH containing similar infomation. It also can provide a file name to write the data of this link. Multiple links may
write their data to the same file:
```cpp
using namespace o2::raw;
RawFileWriter writer { origin };  // origin is o2::header::DataOrigin, will be converted to DAQID and added to RDH.sourceID with RDH version >=6.
auto& lnkA = writer.registerLink(fee_0, cru_0, link_0, endpoint_0, "outfile_0.raw");
auto& lnkB = writer.registerLink(fee_1, cru_0, link_1, endpoint_0, "outfile_0.raw");
..
// or
o2::header::RawDataHeader rdh; // by default, v4 is used currently.
// also o2::header::RDHAny rdh(4);  can be used with the same effect
rdh.feeId = feeX;
rdh.cruID = cruY;
rdh.linkID = linkZ;
rdh.endPointID = endpointQ;
auto& lnkC = writer.registerLink( rdh, "outfile_f.raw");
```

By default the writer will be created for CRU detector (full constructor is `RawFileWriter(o2::header::DataOrigin origin = o2::header::gDataOriginInvalid, bool cru=true)`).
In order to create it for RORC detector use e.g. `RawFileWriter writer { "EMC", false }`. It will automatically impose triggered readout mode and (i) every trigger will be written with its own RDH; (ii) no RDH.stop will be added in the end of the trigger;
(iii) the start of the N-th TimeFrame will be assumed implicitly at the first `trigger_orbit >= SOX_orbit + N*HBFUtils.orbitFirst`, where the `SOX_orbit` is the orbit where SOX flag was set
(`start of continuous` or `start of triggered` mode is set in the beginning with the orbit/bc corresponding to HBFUtils::getFirstIR()).

If needed, user may set manually the non-mutable (for given link) fields of the link RAWDataHeader via direct access to `lnkC.rdhCopy`. This fields will be cloned to all RDHs written for this link.

*   add raw data payload to the `RawFileWriter`, providing the link information and the `o2::InteractionRecord` corresponding to payload.
The data must be provided as a `gsl::span<char>`` and contain only detector GBT (16 bytes words) payload. The annotation by RDH,
formatting to CRU pages and eventual splitting of the large payload to multiple pages will be done by the `RawFileWriter`.
```cpp
writer.addData(cru_0, link_0, endpoint_0, {bc, orbit}, gsl::span( (char*)payload_0, payload_0_size), <optional arguments> );
...
o2::InteractionRecord ir{bc, orbit};
writer.addData(rdh, ir, gsl::span( (char*)payload_f, payload_f_size ), <optional arguments>);
```
where <optional arguments> are currently: `bool preformatted = false, uint32_t trigger = 0, uint32_t detField = 0` (see below for the meaning).

By default the data will be written using o2::header::RAWDataHeader. User can request particular RDH version via `writer.useRDHVersion(v)`.

Note that the `addData` will have no effect if its `{bc, orbit}` exceeds the `HBFUtils.getFirstIR()` by more than `HBFUtils.maxNOrbits`.

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
int carryOverMethod(const o2::header::RDHAny* rdh, const gsl::span<char> data, const char* ptr, int maxSize, int splitID, std::vector<char>& trailer, std::vector<char>& header) const
```

The role of this method is to suggest to the writer how to split the payload: if it was provided to the RawFileWriter using
`writer.setCarryOverCallBack(pointer_on_the_converter_class);` then the `RawFileWriter` will call it before splitting.

It provides to the converter carryOverMethod method the following info:
```cpp
rdh     : RDH of the CRU page being written (pointer to o2::header::RDHAny)
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

The method must return actual size of the bloc which can be written (`<=maxSize`).
If this method populates the trailer, it must ensure that it returns the actual size such that
`actualSize + trailer.size() <= maxSize`.
In case returned `actualSize` is 0, current CRU page will be closed just with user trailer added (if any) and new
query of this method will be done on the new CRU page.

By default, the carry-over callback is not called if remaining data fits to the free space of the 8KB page (or the super-page).
In case the splitting affects the information written in the payload trailer, user may set `writer.setApplyCarryOverToLastPage(true)`.
With this flag set to `ON`, if there was at least one splitting for the user payload provided to `addData` method, then the carry-over
method will be called also for the very last chunk (which by definition does not need splitting) and the supplied trailer will overwrite
the tail of the this chunk instead of adding it incrementally.

Additionally, in case detector wants to add some information between `empty` HBF packet's opening and
closing RDHs (they will be added automatically using the HBFUtils functionality for all HBFs w/o data
between the orbit 0 and last orbit of the TF seen in the simulations), it may implement a callback method
```cpp
void emptyHBFMethod(const o2::header::RDHAny* rdh, std::vector<char>& toAdd) const
```
If such method was registered using `writer.setEmptyPageCallBack(pointer_on_the_converter_class)`, then for every
empty HBF the writer will call it with
```cpp
rdh     : RDH of the CRU page opening empty RDH
toAdd   : a vector (supplied empty) to be filled to a size multipe of 16 bytes
```
The data `toAdd` will be inserted between the star/stop RDHs of the empty HBF.

Adding empty HBF pages for HB's w/o data can be avoided by setting `writer.setDontFillEmptyHBF(true)` before starting conversion. Note that the empty HBFs still will be added for HBs which are supposed to open a new TF.

Some detectors (ITS/MFT) write a special header word after the RDH of every new CRU page (actually, different GBT words for pages w/o and with ``RDH.stop``) in non-empty HBFs. This can be achieved by
another call back method
```cpp
void newRDHMethod(const RDHAny* rdh, bool prevEmpty, std::vector<char>& toAdd) const;
```
It proveds the ``RDH`` of the page for which it is called, information if the previous had some payload and buffer to be filled by the used algorithm.

Some detectors signal the end of the HBF by adding an empty CRU page containing just a header with ``RDH.stop=1`` while others may simply set the ``RDH.stop=1`` in the last CRU page of the HBF (it may appear to be also the 1st and the only page of the HBF and may or may not contain the payload).
This behaviour is steered by ``writer.setAddSeparateHBFStopPage(bool v)`` switch. By default end of the HBF is signaled on separate page.

RawFileWriter will check for every bc/orbit for which at least 1 link was filled by the detector if all other registered links also got `addData` call.
If not, it will be enforced internally with 0 payload and `trigger` and `detField` RDH fields provided in the 1st addData call of this IR.
This may be pose a problem in case detector needs to insert some header even for 0-size payloads.
In this case it is detector's responsibility to make sure that all links receive their proper `addData` call for every IR.

Extra arguments:

* `preformatted` (default: false): The behaviour described above can be modified by providing an extra argument in the `addData` method
```cpp
bool preformatted = true;
writer.addData(cru_0, link_0, endpoint_0, {bc, orbit}, gsl::span( (char*)payload_0, payload_0_size ), preformatted );
...
o2::InteractionRecord ir{bc, orbit};
writer.addData(rdh, ir, gsl::span( (char*)payload_f, payload_f_size ), preformatted );
```

In this case provided span is interpretted as a fully formatted CRU page payload (i.e. it lacks the RDH which will be added by the writer) of the maximum size `8192-sizeof(RDH) = 8128` bytes.
The writer will create a new CRU page with provided payload equipping it with the proper RDH: copying already stored RDH of the current HBF, if the interaction record `ir` belongs to the same HBF or generating new RDH for new HBF otherwise (and filling all missing HBFs in-between). In case the payload size exceeds maximum, an exception will be thrown w/o any attempt to split the page.

* `trigger` (default 0): optionally detector may provide a trigger word for this payload, this word will be OR-ed with the current RDH.triggerType

* `detField` (default 0): optionally detector may provide a custom 32-bit word which will be assigned to RDH.detectorField.

For further details see  ``ITSMFT/common/simulation/MC2RawEncoder`` class and the macro
`Detectors/ITSMFT/ITS/macros/test/run_digi2rawVarPage_its.C` to steer the MC to raw data conversion.

* Update: Use flag HBFUtils.obligatorySOR to start raw data from TF with SOX.

If the HBFUtils.obligatorySOR==false (default) the MC->Raw converted data will start from the 1st TF containing data (i.e. corresponding to HBFUtils.firstOrbitSampled),
the SOX in the RDH will be set only if this TF coincides with the 1st TF of the Run (defined by the HBFUtils.orbitFirst).
With HBFUtils.obligatorySOR==true old behaviour will be preserved: the raw data will start from TF with HBFUtils.orbitFirst with SOX always set and for CRU detectors all HBFs/TFs between HBFUtils.orbitFirst and 1st non-empty HBF will be
filled by dummy RDHs.

## RawFileReader

A class for parsing raw data file(s) with "variable-size" CRU format.
For every encountered link the DPL `SubSpecification` is assigned according to the formula used by DataDistribution:
```cpp
SubSpec = (RDH.cruID<<16) | ((RDH.linkID + 1)<<(RDH.endPointID == 1 ? 8 : 0));
```

This `SubSpecification` is used to define the DPL InputSpecs (and match the OutputSpecs).
An `exception` will be thrown if 2 different link (i.e. with different RDH.feeId) of the same detector will be found to have the same SubSpec (i.e. the same cruID, linkID and PCIe EndPoint).
Also, an `exception` is thrown if the block expected to be a RDH fails to be recognized as RDH.

TEMPORARY UPDATE: given that some detectors (e.g. TOF, TPC) at the moment cannot guarantee a unique (within a given run) mapping FEEID <-> {cruID, linkID, EndPoint},
the `SubSpecification` defined by the DataDistribution may mix data of different links (FEEDs) to single DPL input.
To avoid this, the `SubSpecification` definition is temporarily changed to a 32 bit hash code (Fletcher32). When the new RDHv6 (with source ID) will be put in production, both
DataDistribution and RawFileReader will be able to assign the `SubSpecification` in a consistent way, as a combination of FEEID and SourceID.

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
# for CRU detectors the "readoutCard" record below is optional
# readoutCard = CRU

[input-1]
dataOrigin = TPC
filePath = path_and_name_of_the_data_file1

[input-2]
filePath = path_and_name_of_the_data_file2

[input-1-RORC]
dataOrigin = EMC
filePath = path_and_name_of_the_data_fileX
# for RORC detectors the record below is obligatory
readoutCard = RORC



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
    LOG(info) << "nothing left to read after " << tfID << " TFs read";
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
  --loop arg (=1)                       loop N times (infinite for N<0)
  --min-tf arg (=0)                     min TF ID to process
  --max-tf arg (=4294967295)            max TF ID to process
  --delay arg (=0)                      delay in seconds between consecutive TFs sending
  --buffer-size arg (=1048576)          buffer size for files preprocessing
  --super-page-size arg (=1048576)      super-page size for FMQ parts definition
  --part-per-sp                         FMQ parts per superpage instead of per HBF
  --raw-channel-config arg              optional raw FMQ channel for non-DPL output
  --cache-data                          cache data at 1st reading, may require excessive memory!!!
  --detect-tf0                          autodetect HBFUtils start Orbit/BC from 1st TF seen (at SOX)
  --calculate-tf-start                  calculate TF start from orbit instead of using TType
  --drop-tf arg (=none)                 drop each TFid%(1)==(2) of detector, e.g. ITS,2,4;TPC,4[,0];...
  --start-time arg (=0)                 define TF creation time as start-time + firstTForbit*orbit_duration, ms, otherwise: current time
  --configKeyValues arg                 semicolon separated key=value strings

  # to suppress various error checks / reporting
  --nocheck-packet-increment            ignore /Wrong RDH.packetCounter increment/
  --nocheck-page-increment              ignore /Wrong RDH.pageCnt increment/
  --check-stop-on-page0                 check  /RDH.stop set of 1st HBF page/
  --nocheck-missing-stop                ignore /New HBF starts w/o closing old one/
  --nocheck-starts-with-tf              ignore /Data does not start with TF/HBF/
  --nocheck-hbf-per-tf                  ignore /Number of HBFs per TF not as expected/
  --nocheck-tf-per-link                 ignore /Number of TFs is less than expected/
  --nocheck-hbf-jump                    ignore /Wrong HBF orbit increment/
  --nocheck-no-spage-for-tf             ignore /TF does not start by new superpage/
  --nocheck-no-sox                      ignore /No SOX found on 1st page/
  --nocheck-tf-start-mismatch           ignore /Mismatch between TType-flagged and calculated new TF start/

```

The workflow takes an input from the configuration file (as described in `RawFileReader` section), reads the data and sends them as DPL messages
with the `OutputSpec`s indicated in the configuration file (or defaults). Each link data gets `SubSpecification` according to DataDistribution
scheme.

If `--loop` argument is provided, data will be re-played in loop. The delay (in seconds) can be added between sensding of consecutive TFs to avoid pile-up of TFs. By default at each iteration the data will be again read from the disk.
Using `--cache-data` option one can force caching the data to memory during the 1st reading, this avoiding disk I/O for following iterations, but this option should be used with care as it will eventually create a memory copy of all TFs to read.

At every invocation of the device `processing` callback a full TimeFrame for every link will be added as a multi-part `FairMQ` message and relayed by the relevant channel.
By default each HBF will start a new part in the multipart message. This behaviour can be changed by providing `part-per-sp` option, in which case there will be one part per superpage (Note that this is incompatible to the DPLRawSequencer).

By the default the DataProcessingHeader of each message will have its creation time set to `now()`. This can be changed by passing an option `--configKeyValues "HBFUtils.startTime=<t>"` with `t` being desired run start time in milliseconds: in this case the creation time will be defined as `t + (firstTForbit-HBFUtils.orbitFirst)*orbit_duration` in milliseconds.

The standard use case of this workflow is to provide the input for other worfklows using the piping, e.g.
```cpp
o2-raw-file-reader-workflow --input-conf myConf.cfg | o2-dpl-raw-parser
```
Option `--raw-channel-config <confstring> forces the reader to send all data (single `fair::mq::Parts` containing the whole TF) to raw FairMQ channel, emulating the messages from the DataDistribution.
To inject such a data to DPL one should use a parallel process starting with `o2-dpl-raw-proxy`. An example (note `--session default` added to every executable):

```bash
[Terminal 1]> o2-dpl-raw-proxy --session default -b --dataspec "A:TOF/RAWDATA;B:ITS/RAWDATA;C:MFT/RAWDATA;D:TPC/RAWDATA;E:FT0/RAWDATA" --channel-config "name=readout-proxy,type=pull,method=connect,address=ipc://@rr-to-dpl,transport=shmem,rateLogging=1" | o2-dpl-raw-parser --session default  --input-spec "A:TOF/RAWDATA;B:ITS/RAWDATA;C:MFT/RAWDATA;D:TPC/RAWDATA;E:FT0/RAWDATA"
```

```bash
[Terminal 2]> o2-raw-file-reader-workflow  --session default --loop 1000 --delay 3 --input-conf raw/rawAll.cfg --raw-channel-config "name=raw-reader,type=push,method=bind,address=ipc://@rr-to-dpl,transport=shmem,rateLogging=1" --shm-segment-size 16000000000
```

For testing reason one can request dropping of some fraction of the TFs for particular detector. With option `--drop-tf "ITS,3,2;TPC,10"` the reader will not send the output for ITS in TFs with `(TFid%3)==2` and for TPC in TFs with `(TFid%10)==0`. Note that in order to acknowledge the TF, the message `{FLP/DISTSUBTIMEFRAME/0}` will still be sent even if all detector's data was dropped for a given TF.

## Raw data file checker (standalone executable)

```cpp
Usage:   o2-raw-file-check [options] file0 [... fileN]
Options:
  -h [ --help ]                     print this help message.
  -c [ --input-conf ] arg           read input from configuration file
  -m [ --max-tf] arg (=0xffffffff)  max. TF ID to read (counts from 0)
  -v [ --verbosity ] arg (=0)    1: long report, 2 or 3: print or dump all RDH
  -s [ --spsize ]    arg (=1048576) nominal super-page size in bytes
  --detect-tf0                      autodetect HBFUtils start Orbit/BC from 1st TF seen
  --calculate-tf-start              calculate TF start from orbit instead of using TType
  --rorc                            impose RORC as default detector mode
  --configKeyValues arg             semicolon separated key=value strings
  --nocheck-packet-increment        ignore /Wrong RDH.packetCounter increment/
  --nocheck-page-increment          ignore /Wrong RDH.pageCnt increment/
  --check-stop-on-page0             check  /RDH.stop set of 1st HBF page/
  --nocheck-missing-stop            ignore /New HBF starts w/o closing old one/
  --nocheck-starts-with-tf          ignore /Data does not start with TF/HBF/
  --nocheck-hbf-per-tf              ignore /Number of HBFs per TF not as expected/
  --nocheck-tf-per-link             ignore /Number of TFs is less than expected/
  --nocheck-hbf-jump                ignore /Wrong HBF orbit increment/
  --nocheck-no-spage-for-tf         ignore /TF does not start by new superpage/
  --nocheck-no-sox                  ignore /No SOX found on 1st page/
  --nocheck-tf-start-mismatch       ignore /Mismatch between TType-flagged and calculated new TF start/
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
## DataDistribution TF reader workflow

`o2-raw-tf-reader-workflow` allows to inject into the DPL the `(s)TF` data produced by the DataDistribution, avoiding running DD sTFReader in a separate terminal and sending data via proxies.
The relevant command line options are:
```
--input-data arg
```
input data (obligatory): comma-separated list of input data files and/or files with list of data files and/or directories containing files

```
--max-tf arg (=-1)
```
max TF ID to process (<= 0 : infinite)

```
--loop arg (=0)
```
loop N times (-1 = infinite) over input files (but max-tf has priority if positive)

```
--delay arg (=0)
```
delay in seconds between consecutive TFs sending (provided TFs can be read fast enough)

```
--copy-cmd arg (=XrdSecPROTOCOL=sss,unix xrdcp -N root://eosaliceo2.cern.ch/?src ?dst)
```
copy command for remote files

```
--tf-file-regex arg (=.+\.tf$)
```
regex string to identify TF files: optional to filter data files (if the input contains directories, it will be used to avoid picking non-TF files)

```
--remote-regex arg (=^/eos/aliceo2/.+)
```
regex string to identify remote files (they will be copied to `/tmp/<random>/` directory, then removed after processing)

```
--max-cached-tf arg (=3)
```
max TFs to cache in memory: builds TFs asynchrously to sending, may speed up processing but needs more shmem.

```
--max-cached-files arg (=3)
```
max TF files queued (copied for remote source). For local files almost irrelevant, for remote ones asynchronously creates local copy.

```
--tf-reader-verbosity arg (=0)
```
verbosity level (1 or 2: check RDH, print DH/DPH for 1st or all slices, >2 print RDH)

```
--raw-channel-config arg
```
optional raw FMQ channel for non-DPL output, similar to `o2-raw-file-reader-workflow`.

Since a priory it is not known what kind of data is in the raw TF file provided on the input, by default the reader will define outputs for the raw data of all known detectors (as subspecification-wildcarded `<DET>/RAWDATA`). Additionally, since some detectors might have done processing on the FLP (in which case the DD TF will contain derived data, e.g. cells), the reader will open extra outputs for all kinds of messages for which we know they can be produced on the FLP. Following 3 partially redundant options allow to decrease the number of defined outputs:
```
--onlyDet arg (=all)
```
list of detectors to select from available in the data, outputs for others will not be created, their data will be skipped.
```
--raw-only-det arg (=none)
```
list of detectors for which non-raw outputs (if any) are discarded.
```
--non-raw-only-det arg (=none)
```
list of detectors for which raw outputs are discarded.

The raw data will be propagated (if present) only if the detector is selected in `--onlyDet` and `NOT` selected in `--non-raw-only-det`. The non-raw data will be propagated (if defined for the given detector and present in the file) only if the detector is selected in `--onlyDet` and `NOT` selected in `--raw-only-det`.

## Miscellaneous macros

*   `rawStat.C`: writes into the tree the size per HBF contained in the raw data provided in the RawFileReader config file. No check for synchronization between different links is done.
