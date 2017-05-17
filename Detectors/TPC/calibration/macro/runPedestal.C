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
