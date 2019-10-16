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
#include "TPCCalibration/CalibPedestal.h"
#include "TPCCalibration/CalibRawBase.h"
#endif

void runPedestal(TString fileInfo, TString outputFileName = "", Int_t nevents = 100, Int_t adcMin = 0, Int_t adcMax = 1100, Int_t numberTimeBins = 500, Int_t statisticsType = 0, uint32_t verbosity = 0, uint32_t debugLevel = 0, Int_t firstEvent = 0)
{
  using namespace o2::tpc;
  CalibPedestal ped; //(PadSubset::Region);
  ped.setADCRange(adcMin, adcMax);
  ped.setupContainers(fileInfo, verbosity, debugLevel);
  ped.setStatisticsType(CalibPedestal::StatisticsType(statisticsType));
  ped.setTimeBinRange(0, numberTimeBins);

  //ped.processEvent();
  //ped.resetData();

  CalibRawBase::ProcessStatus status = CalibRawBase::ProcessStatus::Ok;
  //while (ped.processEvent());
  for (Int_t i = firstEvent; i < firstEvent + nevents; ++i) {
    status = ped.processEvent(i);
    cout << "Processing event " << i << " with status " << int(status) << '\n';
    if (status != CalibRawBase::ProcessStatus::Ok) {
      break;
    }
  }
  ped.analyse();

  cout << "Number of processed events: " << ped.getNumberOfProcessedEvents() << '\n';
  cout << "Status: " << int(status) << '\n';
  if (outputFileName.IsNull())
    outputFileName = "Pedestals.root";
  ped.dumpToFile(outputFileName.Data());

  //const CalDet<float>& calPedestal = ped.getPedestal();
  //const CalDet<float>& calNoise    = ped.getNoise();

  //painter::Draw(calPedestal);
  //painter::Draw(calNoise);

  //TCanvas *cPedestal = new TCanvas("cPedestal","Pedestal");
  //auto hPedestal = painter::getHistogram2D(calPedestal.getCalArray(0));
  //hPedestal->SetTitle("Pedestals");
  //hPedestal->Draw("colz");

  //TCanvas *cNoise = new TCanvas("cNoise","Noise");
  //auto hNoise = painter::getHistogram2D(calNoise.getCalArray(0));
  //hNoise->SetTitle("Noise");
  //hNoise->Draw("colz");

  cout << "To display the pedestals run: root.exe $calibMacroDir/drawNoiseAndPedestal.C'(\"" << outputFileName << "\")'\n";
}
