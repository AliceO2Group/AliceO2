/// \file plot_dig_phos.C
/// \brief Simple macro to plot PHOS digits per event

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>

#include <TStopwatch.h>
#include "TCanvas.h"
#include "TH2.h"
//#include "DataFormatsParameters/GRPObject.h"
#include "FairFileSource.h"
#include "FairLogger.h"
#include "FairRunAna.h"
//#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include "PHOSBase/Digit.h"
#include "PHOSBase/Geometry.h"
#include "PHOSSimulation/DigitizerTask.h"
#endif

void plot_dig_phos(int ievent = 0, TString inputfile = "o2dig.root")
{

  TFile* file1 = TFile::Open(inputfile.Data());
  TTree* digTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::phos::Digit>* mDigitsArray = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  digTree->SetBranchAddress("PHSDigit", &mDigitsArray);
  digTree->SetBranchAddress("PHSDigitMCTruth", &labels);

  if (!mDigitsArray) {
    cout << "PHOS digits not registered in the FairRootManager. Exiting ..." << endl;
    return;
  }
  digTree->GetEvent(ievent);

  TH2D* vMod[5][100] = { 0 };
  int primLabels[5][100];
  for (int mod = 1; mod < 5; mod++)
    for (int j = 0; j < 100; j++)
      primLabels[mod][j] = -999;

  o2::phos::Geometry* geom = new o2::phos::Geometry("PHOS");

  std::vector<o2::phos::Digit>::const_iterator it;
  int relId[3];

  for (it = mDigitsArray->begin(); it != mDigitsArray->end(); it++) {
    int absId = (*it).getAbsId();
    double en = (*it).getAmplitude();
    int lab = (*it).getLabel();
    geom->AbsToRelNumbering(absId, relId);
    // check, if this label already exist
    int j = 0;
    bool found = false;
    while (primLabels[relId[0]][j] >= -2) {
      if (primLabels[relId[0]][j] == lab) {
        found = true;
        break;
      } else {
        j++;
      }
    }
    if (!found) {
      primLabels[relId[0]][j] = lab;
    }
    if (!vMod[relId[0]][j]) {
      gROOT->cd();
      vMod[relId[0]][j] =
        new TH2D(Form("hMod%d_prim%d", relId[0], j), Form("hMod%d_prim%d", relId[0], j), 64, 0., 64., 56, 0., 56.);
    }
    vMod[relId[0]][j]->Fill(relId[1] - 0.5, relId[2] - 0.5, en);
  }

  TCanvas* c[5];
  TH2D* box = new TH2D("box", "PHOS module", 64, 0., 64., 56, 0., 56.);
  for (int mod = 1; mod < 5; mod++) {
    c[mod] =
      new TCanvas(Form("DigitInMod%d", mod), Form("PHOS hits in module %d", mod), 10 * mod, 0, 600 + 10 * mod, 400);
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
  }
}