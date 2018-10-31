#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "FITSimulation/Detector.h"
#include "FITBase/Digit.h"
#include "FITReconstruction/RecPoints.h"
#include <TH2F.h>
#endif
void readRecPoints()
{
  using namespace o2::fit;
  //  using namespace o2::fit::Digit;

  // Create histograms
  TDirectory* cwd = gDirectory;
  gDirectory = 0x0;

  TH2F* hMultRec = new TH2F("hMultRec", "photons Recits ", 210, 0, 210, 500, 0, 1000);
  TH2F* hTimeRec = new TH2F("hTimeRec", "Time Recits", 210, 0, 210, 1000, 1, 13);
  TH1F* ht0AC = new TH1F("hT0AC", "T0AC", 100, -1, 1);

  gDirectory = cwd;

  TFile* frec = TFile::Open("o2reco_fit.root");
  std::cout << " Open rec file " << std::endl;
  TTree* recTree = (TTree*)frec->Get("o2sim");
  o2::fit::RecPoints* recArr = new RecPoints;
  recTree->SetBranchAddress("FITRecPoints", &recArr);
  Int_t nevD = recTree->GetEntries(); // recits in cont. readout may be grouped as few events per entry
  std::cout << "Found " << nevD << " events with recits " << std::endl;
  Float_t cfd[208], amp[208];
  for (Int_t iev = 0; iev < nevD; iev++) {
    recTree->GetEvent(iev);

    Float_t t0AC = recArr->GetCollisionTime(0);
    Float_t eventtime = recArr->GetTimeFromDigit();
    ht0AC->Fill(t0AC - eventtime);
    std::cout << iev << " AC " << recArr->GetCollisionTime(0) << " " << eventtime << std::endl;
    for (int ii = 0; ii < 208; ii++) {
      cfd[ii] = amp[ii] = 0;
    }
    for (const auto& d : recArr->getChDgData()) {
      Int_t mcp = d.ChId;
      cfd[mcp] = d.CFDTime;
      amp[mcp] = d.QTCAmpl;
      //  cout<<iev<<" "<<mcp<<" "<< cfd[mcp]<<" "<< amp[mcp]<<endl;
      hMultRec->Fill(Float_t(mcp), amp[mcp]);
      hTimeRec->Fill(Float_t(mcp), cfd[mcp]);
    }
  }
  TFile* Hfile = new TFile("FigFit_rec_pp.root", "RECREATE");
  printf("Writting histograms to root file \n");
  Hfile->cd();
  //Create a canvas, set the view range, show histograms
  //  TCanvas *c1 = new TCanvas("c1","Alice T0 Time ",400,10,600,600);
  hTimeRec->Write();
  hMultRec->Write();
  ht0AC->Write();

} // end of macro
