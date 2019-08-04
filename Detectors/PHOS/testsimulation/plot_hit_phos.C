/// \file plot_hit_phos.C
/// \brief Simple macro to plot PHOS hits per event

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

#include "PHOSBase/Hit.h"
#include "PHOSBase/Geometry.h"
#endif

void plot_hit_phos(int ievent = 0, TString inputfile = "AliceO2_TGeant3.phos.mc_10_event.root")
{
  // macros to plot PHOS hits

  TFile* file1 = TFile::Open(inputfile.Data());
  TTree* hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::phos::Hit>* mHitsArray = nullptr;
  //  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  hitTree->SetBranchAddress("PHSHit", &mHitsArray);
  //  digTree->SetBranchAddress("PHSDigitMCTruth", &labels);

  if (!mHitsArray) {
    cout << "PHOS hits not registered in the FairRootManager. Exiting ..." << endl;
    return;
  }
  hitTree->GetEvent(ievent);

  TH2D* vMod[5][100] = {0};
  int primLabels[5][100];
  for (int mod = 1; mod < 5; mod++)
    for (int j = 0; j < 100; j++)
      primLabels[mod][j] = -1;

  o2::phos::Geometry* geom = new o2::phos::Geometry("PHOS");

  std::vector<o2::phos::Hit>::iterator it;
  int relId[3];

  //  for(it=mHitsArray->begin(); it!=mHitsArray->end(); it++){
  for (auto& it : *mHitsArray) {
    int absId = it.GetDetectorID();
    double en = it.GetEnergyLoss();
    int lab = it.GetTrackID();
    geom->AbsToRelNumbering(absId, relId);
    // check, if this label already exist
    int j = 0;
    bool found = false;
    while (primLabels[relId[0]][j] >= 0) {
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
  for (int mod = 1; mod < 5; mod++) {
    c[mod] =
      new TCanvas(Form("HitInMod%d", mod), Form("PHOS hits in module %d", mod), 10 * mod, 0, 600 + 10 * mod, 400);
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