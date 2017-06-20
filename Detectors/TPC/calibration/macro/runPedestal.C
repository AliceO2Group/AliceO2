// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

void runPedestal(TString fileInfo, TString outputFileName="", Int_t nevents=100, Int_t adcMin=0, Int_t adcMax=1100)
{
  using namespace o2::TPC;
  CalibPedestal ped;//(PadSubset::Region);
  ped.setADCRange(adcMin, adcMax);
  ped.setupContainers(fileInfo);

  ped.ProcessEvent();
  ped.resetData();

  //while (ped.ProcessEvent());
  for (Int_t i=0; i<nevents; ++i) {
    if (ped.ProcessEvent() != CalibRawBase::ProcessStatus::Ok) break;
  }
  ped.analyse();

  cout << "Number of processed events: " << ped.getNumberOfProcessedEvents() << '\n';
  if (outputFileName.IsNull()) outputFileName="Pedestals.root";
  ped.dumpToFile(outputFileName);

  //const CalDet<float>& calPedestal = ped.getPedestal();
  //const CalDet<float>& calNoise    = ped.getNoise();

  //Painter::Draw(calPedestal);
  //Painter::Draw(calNoise);

  //TCanvas *cPedestal = new TCanvas("cPedestal","Pedestal");
  //auto hPedestal = Painter::getHistogram2D(calPedestal.getCalArray(0));
  //hPedestal->SetTitle("Pedestals");
  //hPedestal->Draw("colz");

  //TCanvas *cNoise = new TCanvas("cNoise","Noise");
  //auto hNoise = Painter::getHistogram2D(calNoise.getCalArray(0));
  //hNoise->SetTitle("Noise");
  //hNoise->Draw("colz");

  cout << "To display the pedestals run: root.exe $calibMacroDir/drawNoiseAndPedestal.C'(\"" << outputFileName << "\")'\n";


}
