#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>
#include <iostream>

#include "TROOT.h"
#include <TStopwatch.h>
#include "TCanvas.h"
#include "TH2.h"
#include "TH1.h"
//#include "DataFormatsParameters/GRPObject.h"
#include "FairFileSource.h"
#include <fairlogger/Logger.h>
#include "FairRunAna.h"
//#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include "DataFormatsCPV/Digit.h"
#include "CPVBase/Geometry.h"
#endif

void plot_dig_cpv(int ievent = 0, std::string inputfile = "o2dig.root")
{
  // macros to plot CPV digits

  // Digits
  TFile* file0 = TFile::Open(inputfile.data());
  std::cout << " Open digits file " << inputfile << std::endl;
  TTree* hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::cpv::Digit>* mDigitsArray = nullptr;
  hitTree->SetBranchAddress("CPVDigit", &mDigitsArray);

  if (!mDigitsArray) {
    std::cout << "CPV digits not found in the file. Exiting ..." << std::endl;
    return;
  }
  hitTree->GetEvent(ievent);

  TH1F* hDigitAmplitude[5];
  TH2D* vMod[5][1000] = {0};
  int primLabels[5][1000];
  for (int mod = 1; mod < 5; mod++) {
    hDigitAmplitude[mod] = new TH1F(Form("hDigitAmplitude_mod%d", mod),
                                    Form("Digit amplitudes in module %d", mod),
                                    4096, 0., 4096.);
    for (int j = 0; j < 100; j++)
      primLabels[mod][j] = -1;
  }

  std::vector<o2::cpv::Digit>::const_iterator it;
  short relId[3];
  std::cout << "I start digit cycling" << std::endl;

  for (it = mDigitsArray->begin(); it != mDigitsArray->end(); it++) {
    short absId = (*it).getAbsId();
    float en = (*it).getAmplitude();
    int lab = (*it).getLabel(); //TODO
    //std::cout << "label = " << lab << std::endl;
    o2::cpv::Geometry::absToRelNumbering(absId, relId);
    hDigitAmplitude[relId[0]]->Fill(en);
    // check, if this label already exist
    int j = 0;
    bool found = false;
    //no labels for the time being
    // while (primLabels[relId[0]][j] >= -2) {
    //   if (primLabels[relId[0]][j] == lab) {
    // 	found = true;
    // 	break;
    //   } else {
    // 	j++;
    //   }
    // }
    if (!found) {
      primLabels[relId[0]][j] = lab;
    }
    if (!vMod[relId[0]][j]) {
      gROOT->cd();
      vMod[relId[0]][j] =
        new TH2D(Form("hMod%d_prim%d", relId[0], j), Form("hMod%d_prim%d", relId[0], j), 128, 0., 128., 60, 0., 60.);
    }
    vMod[relId[0]][j]->Fill(relId[1] - 0.5, relId[2] - 0.5, en);
  }

  std::cout << "I finish cycling digits" << std::endl;

  TCanvas *c[5], *c_ampl[5];
  TH2D* box = new TH2D("box", "CPV module", 128, 0., 128., 60, 0., 60.);
  for (int mod = 2; mod < 5; mod++) {
    c[mod] =
      new TCanvas(Form("DigitInMod%d", mod), Form("CPV digits in module %d", mod), 10 * mod, 0, 600 + 10 * mod, 400);
    box->Draw();
    int j = 0;
    while (vMod[mod][j]) {
      vMod[mod][j]->SetLineColor(j + 1);
      if (j == 0)
        vMod[mod][j]->Draw("box");
      else
        vMod[mod][j]->Draw("boxsame");
      j++;
    }
    c_ampl[mod] = new TCanvas(Form("DigitAmplitudes_%d", mod), Form("DigitAmplitudes_%d", mod),
                              10 * mod + 800, 0, 600 + 10 * mod + 200, 400);
    hDigitAmplitude[mod]->Draw();
  }
}
