#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>

#include "TROOT.h"
#include <TStopwatch.h>
#include "TCanvas.h"
#include "TH2.h"
//#include "DataFormatsParameters/GRPObject.h"
#include "FairFileSource.h"
#include <fairlogger/Logger.h>
#include "FairRunAna.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CPVBase/Geometry.h"
#include "DataFormatsCPV/Hit.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DetectorsCommonDataFormats/DetID.h"
#endif

using namespace o2::detectors;

void plot_hit_cpv(int ievent = 0, std::string inputprefix = "o2sim")
{
  // macros to plot CPV hits

  // Hits
  std::string inputfile(o2::base::DetectorNameConf::getHitsFileName(DetID::CPV, inputprefix));
  TFile* file0 = TFile::Open(inputfile.c_str());
  std::cout << " Open hits file " << inputfile << std::endl;
  TTree* hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::cpv::Hit>* mHitsArray = nullptr;
  hitTree->SetBranchAddress("CPVHit", &mHitsArray);

  if (!mHitsArray) {
    std::cout << "CPV hits not registered in the FairRootManager. Exiting ..." << std::endl;
    return;
  }
  hitTree->GetEvent(ievent);

  TH2D* vMod[5][100] = {0};
  int primLabels[5][100];
  for (int mod = 1; mod < 5; mod++)
    for (int j = 0; j < 100; j++)
      primLabels[mod][j] = -1;

  std::vector<o2::cpv::Hit>::iterator it;
  short relId[3];

  //  for(it=mHitsArray->begin(); it!=mHitsArray->end(); it++){

  for (auto& it : *mHitsArray) {
    short absId = it.GetDetectorID();
    float en = it.GetEnergyLoss();
    int lab = it.GetTrackID();
    o2::cpv::Geometry::absToRelNumbering(absId, relId);
    printf("reldId=(%d,%d,%d) \n", relId[0], relId[1], relId[2]);
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
        new TH2D(Form("hMod%d_prim%d", relId[0], j), Form("hMod%d_prim%d", relId[0], j), 128, 0., 128., 60, 0., 60.);
    }
    vMod[relId[0]][j]->Fill(relId[1] - 0.5, relId[2] - 0.5, en);
  }

  TCanvas* c[5];
  for (int mod = 1; mod < 5; mod++) {
    c[mod] =
      new TCanvas(Form("HitInMod%d", mod), Form("CPV hits in module %d", mod), 10 * mod, 0, 600 + 10 * mod, 400);
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
