//#if (!defined(__CINT__) && !defined(__CLING__)) || defined(__MAKECINT__)
#include "TObjString.h"
#include "TObjArray.h"
#include "FairLogger.h"

#include "TPCBase/PadPos.h"

#include "TPCReconstruction/RawReader.h"

#include <iostream>
#include <fstream>
#include <memory>
using namespace std;
using namespace o2::TPC;
//#endif
/*
.L RunSimpleEventDisplay.C+
RunCompareMode3("GBTx0_Run005:0:0;GBTx1_Run005:1:0")
*/




//__________________________________________________________________________
void RunCompareMode3(TString fileInfo)
{
  FairLogger *logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("DEBUG");

  auto arrData = fileInfo.Tokenize("; ");
  for (auto o : *arrData) {
    const TString& data = static_cast<TObjString*>(o)->String();
    LOG(INFO) << "Checking file " << data.Data() << FairLogger::endl;
    // get file info: file name, cru, link
    RawReader rawReaderRaw;
    RawReader rawReaderDec;
    rawReaderRaw.setUseRawInMode3(true);
    rawReaderDec.setUseRawInMode3(false);
    rawReaderRaw.addInputFile(data.Data());
    rawReaderDec.addInputFile(data.Data());

    auto eventInfoVecRaw = rawReaderRaw.getEventInfo(rawReaderRaw.getFirstEvent());
    auto eventInfoVecDec = rawReaderDec.getEventInfo(rawReaderDec.getFirstEvent());

    if (eventInfoVecRaw->begin()->header.dataType != 3) {
      LOG(ERROR) << "Readout mode was " << (int) eventInfoVecRaw->begin()->header.dataType << " instead of 3." << FairLogger::endl;
      return;
    }
    std::cout << eventInfoVecRaw->begin()->path << " "
      << eventInfoVecRaw->begin()->posInFile << " "
      << eventInfoVecRaw->begin()->region << " "
      << eventInfoVecRaw->begin()->link << " "
      << (int)eventInfoVecRaw->begin()->header.dataType << " "
      << (int)eventInfoVecRaw->begin()->header.headerVersion << " "
      << eventInfoVecRaw->begin()->header.nWords << " "
      << eventInfoVecRaw->begin()->header.timeStamp() << " "
      << eventInfoVecRaw->begin()->header.eventCount() << std::endl;
    std::cout << eventInfoVecDec->begin()->path << " "
      << eventInfoVecDec->begin()->posInFile << " "
      << eventInfoVecDec->begin()->region << " "
      << eventInfoVecDec->begin()->link << " "
      << (int)eventInfoVecDec->begin()->header.dataType << " "
      << (int)eventInfoVecDec->begin()->header.headerVersion << " "
      << eventInfoVecDec->begin()->header.nWords << " "
      << eventInfoVecDec->begin()->header.timeStamp() << " "
      << eventInfoVecDec->begin()->header.eventCount() << std::endl;


    // first event contains SYNC Pattern 

    for(uint64_t ev = rawReaderRaw.getFirstEvent()+1; ev <= rawReaderRaw.getLastEvent(); ++ev) {
      if (rawReaderRaw.loadEvent(ev) != rawReaderDec.loadEvent(ev)) {
        LOG(ERROR) << "Event " << ev << " can't be decoded by both decoders" << FairLogger::endl;
        return;
      }

      PadPos padPos;
      while (std::shared_ptr<std::vector<uint16_t>> dataRaw = rawReaderRaw.getNextData(padPos)) {
        std::shared_ptr<std::vector<uint16_t>> dataDec = rawReaderDec.getData(padPos);
        if ( (*dataRaw) != (*dataDec) ){
          LOG(ERROR) << "Data is not equal" << FairLogger:: endl;
          std::cout << "data 1:" << std::endl;
          for (auto &a : *dataRaw) {
            std::cout << a << std::endl;
          }
          std::cout << "data 2:" << std::endl;
          for (auto &a : *dataDec) {
            std::cout << a << std::endl;
          }
          std::cout << std::endl;
        }
      }
    }
  }
}

