<!-- doxy
\page refFITraw FIT raw
/doxy -->

# DataBlockBase
DataBlockWrapper - wrapper for raw data structures\
There should be three static fields in raw data structs, which defines its "signature":\
  payloadSize - actual payload size per one raw data struct element (can be larger than GBTword size!)\
  payloadPerGBTword - maximum payload per one GBT word\
  MaxNelements - maximum number of elements per data block(for header it should be equal to 1)\
  MinNelements - minimum number of elements per data block(for header it should be equal to 1)\

Also it requares several methods:\
  print() - for printing raw data structs\
  getIntRec() - for InteractionRecord extraction, should be in Header struct\

DataBlockBase - base class for making composition of raw data structures, uses CRTP(static polyporphism)\
usage:\
  class DataBlockOfYourModule: public DataBlockBase< DataBlockOfYourModule, RawHeaderStruct, RawDataStruct ...>\
  define "deserialization" method with deserialization logic for current DataBlock\
  define "sanityCheck" method for checking if the DataBlock is correct\
Warning! Classes should be simple, without refs and pointers!\
Example:
  /alice/O2/Detectors/FIT/FT0/raw/include/FT0Raw/DataBlockFT0.h
# DigitBlockBase
Base class for digit block, which containes "digit" structures\
usage:\
  class DigitBlockYourDetector : public DigitBlockBase<DigitBlockYourDetector>\
  define "template <class DataBlockType> void processDigits(DataBlockType& dataBlock, int linkID)" with Raw2Digit conversion logic\
  define "template <class DigitType,...>void getDigits(std::vector<DigitType>&... vecDigits) for pushing digits into vectors which will be pushed through DPL channels,
    variadic arguments are applied\
Example:\
  /alice/O2/Detectors/FIT/FT0/raw/include/FT0Raw/DigitBlockFT0.h 

# RawReaderBase
Base class for raw reader.\
usage:\
  class RawReaderFT0Base : public RawReaderBase<DigitBlockYourDetector>\
  just provide statically which linkID corresponds to DataBlockOfYourModule\
Example:\
  /alice/O2/Detectors/FIT/FT0/raw/include/FT0Raw/RawReaderFT0Base.h\
  
# RawReader 
Stores vectors with digit classes and manages them at DPL level\
Example: \
  /alice/O2/Detectors/FIT/FT0/workflow/include/FT0Workflow/RawReaderFT0