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
#include <iostream>
#include <vector>
#include <string_view>
#include "TString.h"
#include "TPCCalibration/DigitDump.h"
#endif

void dumpDigits(std::vector<std::string_view> fileInfos, TString outputFileName = "", int nevents = 100,
                int adcMin = -100, int adcMax = 1100,
                int firstTimeBin = 0, int lastTimeBin = 1000,
                float noiseThreshold = -1,
                TString pedestalAndNoiseFile = "",
                uint32_t verbosity = 0, uint32_t debugLevel = 0)
{
  using namespace o2::tpc;
  DigitDump dig;
  dig.setDigitFileName(outputFileName.Data());
  dig.setPedestalAndNoiseFile(pedestalAndNoiseFile.Data());
  dig.setADCRange(adcMin, adcMax);
  dig.setTimeBinRange(firstTimeBin, lastTimeBin);
  dig.setNoiseThreshold(noiseThreshold);

  CalibRawBase::ProcessStatus status = CalibRawBase::ProcessStatus::Ok;
  for (const auto& fileInfo : fileInfos) {
    dig.setupContainers(fileInfo.data(), verbosity, debugLevel);

    for (int i = 0; i < nevents; ++i) {
      status = dig.processEvent(i);
      cout << "Processing event " << i << " with status " << int(status) << '\n';
      if (status == CalibRawBase::ProcessStatus::IncompleteEvent) {
        continue;
      } else if (status != CalibRawBase::ProcessStatus::Ok) {
        break;
      }
    }
  }
}
