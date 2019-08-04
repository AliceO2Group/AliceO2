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

#include "CPVBase/Digit.h"
#include "CPVBase/Geometry.h"
#include "CPVSimulation/DigitizerTask.h"
#endif

void plot_dig_cpv(int ievent = 0, std::string inputfile = "o2dig.root")
{
  // macros to plot CPV hits

  // Hits
  TFile* file0 = TFile::Open("o2dig.root");
  std::cout << " Open hits file " << inputfile << std::endl;
  TTree* hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::cpv::Digit>* mDigitsArray = nullptr;
  hitTree->SetBranchAddress("CPVHit", &mDigitsArray);

  if (!mDigitsArray) {
    cout << "CPV digits not found in the file. Exiting ..." << endl;
    return;
  }
  hitTree->GetEvent(ievent);

  TH2D* vMod[5][100] = {0};
  int primLabels[5][100];
  for (int mod = 1; mod < 5; mod++)
    for (int j = 0; j < 100; j++)
      primLabels[mod][j] = -1;

  o2::cpv::Geometry* geom = new o2::cpv::Geometry("CPVRun3");

  std::vector<o2::cpv::Digit>::const_iterator it;
  int relId[3];

  for (it = mDigitsArray->begin(); it != mDigitsArray->end(); it++) {
    int absId = (*it).getAbsId();
    double en = (*it).getAmplitude();
    int lab = (*it).getLabel(0);
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
        new TH2D(Form("hMod%d_prim%d", relId[0], j), Form("hMod%d_prim%d", relId[0], j), 60, 0., 60., 128, 0., 128.);
    }
    vMod[relId[0]][j]->Fill(relId[1] - 0.5, relId[2] - 0.5, en);
  }

  TCanvas* c[5];
  TH2D* box = new TH2D("box", "CPV module", 60, 0., 60., 128, 0., 128.);
  for (int mod = 1; mod < 5; mod++) {
    c[mod] =
      new TCanvas(Form("DigitInMod%d", mod), Form("CPV hits in module %d", mod), 10 * mod, 0, 600 + 10 * mod, 400);
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
