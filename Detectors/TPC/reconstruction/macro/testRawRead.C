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
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"

#include "TPCReconstruction/GBTFrameContainer.h"
#endif

using namespace o2::TPC;
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

  std::vector<Digit> digits(80);

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
