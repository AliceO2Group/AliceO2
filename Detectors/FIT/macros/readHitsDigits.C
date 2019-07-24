#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/HitType.h"
#include <TH2F.h>
#include <TTree.h>
#include <TFile.h>

void readHitsDigits()
{

  // Create histograms
  TDirectory* cwd = gDirectory;
  gDirectory = 0x0;

  TH2F* hMultHit = new TH2F("hMultHits", "photons Hits ", 210, 0, 210, 500, 0, 5000);
  TH2F* hTimeHitA = new TH2F("hTimeAhit", "Time Hits", 210, 0, 210, 1000, 11, 12);
  TH2F* hTimeHitC = new TH2F("hTimeChit", "Time Hits", 210, 0, 210, 1000, 2.8, 3.8);
  TH2F* hMultDig = new TH2F("hMultDig", "photons Digits ", 210, 0, 210, 500, 0, 20);
  TH2F* hTimeDig = new TH2F("hTimeDig", "Time Digits", 210, 0, 210, 100, -1, 1);

  gDirectory = cwd;

  TFile* fhit = new TFile("o2sim.root");
  TTree* hitTree = (TTree*)fhit->Get("o2sim");
  std::vector<o2::ft0::HitType>* hitArray = nullptr;
  hitTree->SetBranchAddress("FT0Hit", &hitArray);
  Int_t nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  // std::cout << "Found " << nevH << " events with hits " << std::endl;

  Double_t hit_time[240];
  Int_t countE[240];
  // Event ------------------------- LOOP
  for (Int_t ievent = 0; ievent < nevH; ievent++) {
    hitTree->GetEntry(ievent);
    for (int ii = 0; ii < 240; ii++) {
      countE[ii] = 0;
      hit_time[ii] = 0;
    }
    for (auto& hit : *hitArray) {
      Int_t detID = hit.GetDetectorID();
      hit_time[detID] = hit.GetTime();
      hTimeHitA->Fill(detID, hit_time[detID]);
      hTimeHitC->Fill(detID, hit_time[detID]);
      countE[detID]++;
    }
    for (int ii = 0; ii < 208; ii++) {
      if (countE[ii] > 100) {
        hMultHit->Fill(ii, countE[ii]);
        //	std::cout<<ii<<" "<<countE[ii]<<endl;
      }
    }
  }

  TFile* fdig = TFile::Open("ft0digits.root");
  std::cout << " Open digits file " << std::endl;
  TTree* digTree = (TTree*)fdig->Get("o2sim");
  std::vector<o2::ft0::Digit>* digArr = new std::vector<o2::ft0::Digit>;
  digTree->SetBranchAddress("FT0Digit", &digArr);
  Int_t nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  // std::cout << "Found " << nevD << " events with digits " << std::endl;
  Float_t cfd[208], amp[208], part[208];
  for (Int_t iev = 0; iev < nevD; iev++) {
    digTree->GetEvent(iev);
    for (const auto& digit : *digArr) {
      for (int ii = 0; ii < 208; ii++) {
        cfd[ii] = amp[ii] = 0;
      }
      Double_t evtime = digit.getTime();
      for (const auto& d : digit.getChDgData()) {
        Int_t mcp = d.ChId;
        cfd[mcp] = d.CFDTime - evtime - 12.5;
        amp[mcp] = d.QTCAmpl;
        part[mcp] = d.numberOfParticles;
        //     cout << iev << " " << mcp << " " << cfd[mcp] << " " << amp[mcp] << " " << part[mcp] << endl;
        hMultDig->Fill(Float_t(mcp), amp[mcp]);
        hTimeDig->Fill(Float_t(mcp), cfd[mcp]);
      }
    }
  }
  TFile* Hfile = new TFile("FigFit_hits_pp.root", "RECREATE");
  printf("Writting histograms to root file \n");
  Hfile->cd();
  //Create a canvas, set the view range, show histograms
  //  TCanvas *c1 = new TCanvas("c1","Alice T0 Time ",400,10,600,600);
  hTimeHitA->Write();
  hTimeHitC->Write();
  hMultHit->Write();
  hTimeDig->Write();
  hMultDig->Write();

} // end of macro
#endif
