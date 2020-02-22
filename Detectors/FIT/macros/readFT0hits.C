#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/HitType.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include <TFile.h>
#include <TH2F.h>
#include <TTree.h>

void readFT0hits()
{

  // Create histograms
  TDirectory* cwd = gDirectory;
  gDirectory = 0x0;

  TH2F* hMultHit =
    new TH2F("hMultHits", "photons Hits ", 210, 0, 210, 500, 0, 5000);
  TH2F* hTimeHitA = new TH2F("hTimeAhit", "Time Hits", 210, 0, 210, 500, -1, 1);
  TH2F* hTimeHitC = new TH2F("hTimeChit", "Time Hits", 210, 0, 210, 200, -1, 1);
  TH2F* hMultDig =
    new TH2F("hMultDig", "amplitude  ", 210, 0, 210, 500, 0, 1000);
  TH2F* hPel = new TH2F("hPelDig", "N p.e. ", 210, 0, 210, 500, 0, 10000);
  TH2F* hXYA = new TH2F("hXYA", "X vs Y A side", 400, -20, 20, 400, -20, 20);
  TH2F* hXYC = new TH2F("hXYC", "X vs Y C side", 400, -20, 20, 400, -20, 20);

  gDirectory = cwd;

  TFile* fhit = new TFile("o2sim.root");
  TTree* hitTree = (TTree*)fhit->Get("o2sim");

  o2::dataformats::MCEventHeader* mcHeader = nullptr;
  /*  if (!hitTree->GetBranch("MCEventHeader.")) {
    LOG(FATAL) << "Did not find MC event header in the input header file." <<
    FairLogger::endl;
    }*/
  hitTree->SetBranchAddress("MCEventHeader.", &mcHeader);

  std::vector<o2::ft0::HitType>* hitArray = nullptr;
  hitTree->SetBranchAddress("FT0Hit", &hitArray);
  Int_t nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  // std::cout << "Found " << nevH << " events with hits " << std::endl;

  Double_t hit_time[240];
  Int_t countE[240];
  // Event ------------------------- LOOP
  for (Int_t ievent = 0; ievent < nevH; ievent++) {
    hitTree->GetEntry(ievent);
    ///* std::cout<<ievent<<" b"<<mcHeader->GetB()<<std::endl;
    for (int ii = 0; ii < 240; ii++) {
      countE[ii] = 0;
      hit_time[ii] = 0;
    }
    for (auto& hit : *hitArray) {
      Int_t detID = hit.GetDetectorID();
      hit_time[detID] = hit.GetTime();
      hTimeHitA->Fill(detID, hit_time[detID] - 11.04);
      hTimeHitC->Fill(detID, hit_time[detID] - 2.91);
      countE[detID]++;
      if (detID < 97)
        hXYA->Fill(hit.GetX(), hit.GetY());
      if (detID > 96)
        hXYC->Fill(hit.GetX(), hit.GetY());
    }
    for (int ii = 0; ii < 208; ii++) {
      if (countE[ii] > 100) {
        hMultHit->Fill(ii, countE[ii]);
        //	std::cout<<ii<<" "<<countE[ii]<<endl;
      }
    }
  }
  TFile* Hfile = new TFile("ft0_hits_pp.root", "RECREATE");
  printf("Writting histograms to root file \n");
  Hfile->cd();
  // Create a canvas, set the view range, show histograms
  //  TCanvas *c1 = new TCanvas("c1","Alice T0 Time ",400,10,600,600);
  hTimeHitA->Scale(1. / 250.);
  hTimeHitC->Scale(1. / 250.);
  hTimeHitA->Write();
  hTimeHitC->Write();
  hMultHit->Write();
  hXYA->Write();
  hXYC->Write();

} // end of macro
#endif
