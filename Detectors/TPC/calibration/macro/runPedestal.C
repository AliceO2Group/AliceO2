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
#include "TFile.h"
#include "TPCCalibration/CalibPedestal.h"
#include "TPCCalibration/CalibRawBase.h"
#endif

void runPedestal(std::vector<std::string_view> fileInfos, TString outputFileName = "", Int_t nevents = 100, Int_t adcMin = 0, Int_t adcMax = 1100, Int_t firstTimeBin = 0, Int_t lastTimeBin = 450, Int_t statisticsType = 0, uint32_t verbosity = 0, uint32_t debugLevel = 0, Int_t firstEvent = 0)
{
  using namespace o2::tpc;
  CalibPedestal ped; //(PadSubset::Region);
  ped.setADCRange(adcMin, adcMax);
  ped.setStatisticsType(StatisticsType(statisticsType));
  ped.setTimeBinRange(firstTimeBin, lastTimeBin);

  //ped.processEvent();
  //ped.resetData();

  CalibRawBase::ProcessStatus status = CalibRawBase::ProcessStatus::Ok;

  for (const auto& fileInfo : fileInfos) {
    ped.setupContainers(fileInfo.data(), verbosity, debugLevel);

    for (Int_t i = firstEvent; i < firstEvent + nevents; ++i) {
      status = ped.processEvent(i);
      cout << "Processing event " << i << " with status " << int(status) << '\n';
      if (status == CalibRawBase::ProcessStatus::IncompleteEvent) {
        continue;
      } else if (status != CalibRawBase::ProcessStatus::Ok) {
        break;
      }
    }
  }
  ped.analyse();

  std::cout << "Number of processed events: " << ped.getNumberOfProcessedEvents() << '\n';
  std::cout << "Status: " << int(status) << '\n';
  if (outputFileName.IsNull()) {
    outputFileName = "Pedestals.root";
  }
  ped.dumpToFile(outputFileName.Data());

  std::cout << "To display the pedestals run: root.exe $calibMacroDir/drawNoiseAndPedestal.C'(\"" << outputFileName << "\")'\n";
}
