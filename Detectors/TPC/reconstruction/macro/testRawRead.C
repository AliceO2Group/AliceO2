using namespace AliceO2::TPC;
void testRawRead(std::string filename)
{
  TH1F *hDigits = new TH1F("hDigits","digits", 1000, 0., 1000.);
  TH2F *hRows   = new TH2F("hRows","rows", 20, 0., 20., 140, 0., 140.);
  int iSize = 4000000;
  int iCRU = 0;
  int iLink = 0;

  GBTFrameContainer cont(iSize,iCRU,iLink);
  cont.setEnableAdcClockWarning(false);
  cont.setEnableSyncPatternWarning(false);
  cont.setEnableStoreGBTFrames(true);
  cont.setEnableCompileAdcValues(true);

  std::cout << "Read digits from file ...\n";
  cont.addGBTFramesFromBinaryFile(filename);
  std::cout << " ... done. Read " << cont.getSize() << "\n";

  std::vector<DigitData> digits(80);

  int maxPad=-1;

  while (cont.getData(digits)) {
    //for (int i=0; i<4000;++i) {
      //cont.getData(digits);
    for (const auto& digit : digits) {
      hDigits->Fill(digit.getCharge());
      int pad = digit.getPad();
      hRows->Fill(digit.getRow(), pad);
      maxPad = TMath::Max(pad, maxPad);
    }
    digits.clear();
  }

  cout << "MaxPad: " << maxPad << "\n";
  TCanvas *c1 = new TCanvas("c1","c1");
  hDigits->Draw();
  TCanvas *c2 = new TCanvas("c2","c2");
  hRows->Draw("colz");
}
