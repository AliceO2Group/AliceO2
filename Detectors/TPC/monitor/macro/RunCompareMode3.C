// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TObjString.h"
#include "TObjArray.h"
#include "FairLogger.h"

#include "TPCBase/PadPos.h"

#include "TPCReconstruction/RawReader.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#endif

using namespace std;
using namespace o2::tpc;
/*
.L RunSimpleEventDisplay.C+
RunCompareMode3("GBTx0_Run005:0:0;GBTx1_Run005:1:0")
*/

//__________________________________________________________________________
void RunCompareMode3(TString fileInfo)
{
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  auto arrData = fileInfo.Tokenize("; ");
  uint64_t checkedAdcValues = 0;
  for (const auto& o : *arrData) {
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
      LOG(ERROR) << "Readout mode was " << (int)eventInfoVecRaw->begin()->header.dataType << " instead of 3." << FairLogger::endl;
      return;
    }
    //    std::cout << eventInfoVecRaw->begin()->path << " "
    //      << eventInfoVecRaw->begin()->posInFile << " "
    //      << eventInfoVecRaw->begin()->region << " "
    //      << eventInfoVecRaw->begin()->link << " "
    //      << (int)eventInfoVecRaw->begin()->header.dataType << " "
    //      << (int)eventInfoVecRaw->begin()->header.headerVersion << " "
    //      << eventInfoVecRaw->begin()->header.nWords << " "
    //      << eventInfoVecRaw->begin()->header.timeStamp() << " "
    //      << eventInfoVecRaw->begin()->header.eventCount() << std::endl;
    //    std::cout << eventInfoVecDec->begin()->path << " "
    //      << eventInfoVecDec->begin()->posInFile << " "
    //      << eventInfoVecDec->begin()->region << " "
    //      << eventInfoVecDec->begin()->link << " "
    //      << (int)eventInfoVecDec->begin()->header.dataType << " "
    //      << (int)eventInfoVecDec->begin()->header.headerVersion << " "
    //      << eventInfoVecDec->begin()->header.nWords << " "
    //      << eventInfoVecDec->begin()->header.timeStamp() << " "
    //      << eventInfoVecDec->begin()->header.eventCount() << std::endl;

    // first event contains SYNC Pattern

    for (uint64_t ev = rawReaderRaw.getFirstEvent(); ev <= rawReaderRaw.getLastEvent(); ++ev) {
      if (rawReaderRaw.loadEvent(ev) != rawReaderDec.loadEvent(ev)) {
        LOG(ERROR) << "Event " << ev << " can't be decoded by both decoders" << FairLogger::endl;
        return;
      }

      PadPos padPos;
      while (std::shared_ptr<std::vector<uint16_t>> dataRaw = rawReaderRaw.getNextData(padPos)) {
        std::shared_ptr<std::vector<uint16_t>> dataDec = rawReaderDec.getData(padPos);
        if (dataDec->size() - dataRaw->size() == 2) {
          for (int i = 0; i < std::min(dataDec->size(), dataRaw->size()); ++i) {
            ++checkedAdcValues;
            if (dataRaw->at(i) - dataDec->at(i + 1) != 0) {
              LOG(ERROR) << "Data is not equal in event " << ev
                         << " for pad " << (int)padPos.getPad()
                         << " in row " << (int)padPos.getRow()
                         << " in timebin " << i
                         << " RawVec size: " << dataRaw->size()
                         << " DecVec size: " << dataDec->size()
                         << FairLogger::endl;
              LOG(ERROR) << "Raw: " << dataRaw->at(i) << " \tDec: " << dataDec->at(i) << FairLogger::endl;
            }
          }
        } else {
          for (int i = 0; i < std::min(dataDec->size(), dataRaw->size()); ++i) {
            ++checkedAdcValues;
            if (dataRaw->at(i) - dataDec->at(i) != 0) {
              LOG(ERROR) << "Data is not equal in event " << ev
                         << " for pad " << (int)padPos.getPad()
                         << " in row " << (int)padPos.getRow()
                         << " in timebin " << i
                         << " RawVec size: " << dataRaw->size()
                         << " DecVec size: " << dataDec->size()
                         << FairLogger::endl;
              LOG(ERROR) << "Raw: " << dataRaw->at(i) << " \tDec: " << dataDec->at(i) << FairLogger::endl;
            }
          }
        }
      }
    }
    LOG(INFO) << "Compared " << (double)checkedAdcValues / 1000000 << "M ADC values in total" << FairLogger::endl;
  }
  LOG(INFO) << "Compared " << (double)checkedAdcValues / 1000000 << "M ADC values in total" << FairLogger::endl;
}
