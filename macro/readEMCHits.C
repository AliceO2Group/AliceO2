#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TStopwatch.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TH2F.h>
#include <memory>
#include <iostream>
#include "FairLogger.h"
#include "SimulationDataFormat/RunContext.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "EMCALBase/Hit.h"
#endif

/// read and draw the hits for EMC obtained from simulation
void readEMCHits(std::string path = "./",
                 std::string mcfileName = "o2sim.root")
{
  if (path.back() != '/') {
    path += '/';
  }
  TH2* ep = new TH2F("etaph", "hist", 100, -0.7, 0.7, 100, 0., 7.);

  std::unique_ptr<TFile> mcfile(TFile::Open((path + mcfileName).c_str()));
  if (!mcfile || mcfile->IsZombie()) {
    std::cout << "Failed to open input hit file " << (path + mcfileName) << std::endl;
    return;
  }

  TTree* hitTree = (TTree*)mcfile->Get("o2sim");
  if (!hitTree) {
    std::cout << "Failed to get hit tree" << std::endl;
    return;
  }

  std::vector<o2::emcal::Hit>* dv = nullptr;
  hitTree->SetBranchAddress("EMCHit", &dv);

  for (int iev = 0; iev < hitTree->GetEntries(); iev++) {
    hitTree->GetEntry(iev);
    for (const auto& h : *dv) {
      TVector3 posvec(h.GetX(), h.GetY(), h.GetZ());
      ep->Fill(posvec.Eta(), TVector2::Phi_0_2pi(posvec.Phi()));
    }
  }
  ep->Draw("colz");
}
