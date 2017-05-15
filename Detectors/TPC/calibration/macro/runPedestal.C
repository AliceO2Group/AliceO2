void runPedestal(TString fileInfo="GBTx0_Battery_Floating:0:0;GBTx1_Battery_Floating:1:0", TString outputFileName="")
{
  using namespace o2::TPC;
  CalibPedestal ped;//(PadSubset::Region);
  ped.setupContainers(fileInfo);

  //while (ped.ProcessEvent());
  for (Int_t i=0; i<100; ++i) {
    ped.ProcessEvent();
  }
  ped.analyse();

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

  cout << "To display the pedestals run: root.exe $calibMacroDir/drawNoiseAndPedestal.C'(\" " << outputFileName << "\")'\n";


}
