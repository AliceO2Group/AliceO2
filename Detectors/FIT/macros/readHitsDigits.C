#include "FITSimulation/Detector.h"
#include "FITBase/Digit.h"
#include <TH2F.h>
void readHitsDigits() 
{
  using namespace o2::fit;
  //  using namespace o2::fit::Digit;
 
  // Create histograms
  TDirectory* cwd = gDirectory;
  gDirectory = 0x0;
 
  TH2F *hMultHit = new TH2F("hMultHits","photons Hits ",210, 0, 210, 500, 0, 5000);
  TH2F *hTimeHit = new TH2F("hTimeAChit", "Time Hits", 210, 0, 210, 1000, 0, 15);  
  TH2F *hMultDig = new TH2F("hMultDig","photons Digits ",210, 0, 210, 500, 0, 20);
  TH2F *hTimeDig = new TH2F("hTimeDig", "Time Digits", 210, 0, 210, 300, 0, 15);  

  gDirectory = cwd; 
  
  TFile *fhit = new TFile("o2sim.root");
  TTree *hitTree = (TTree*)fhit->Get("o2sim");
  std::vector<o2::fit::HitType>* hitArray = nullptr;
  hitTree->SetBranchAddress("FITHit", &hitArray);
  Int_t nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  std::cout << "Found " << nevH << " events with hits " << std::endl;
  
  Double_t hit_time[240];
  Int_t countE[240];
   // Event ------------------------- LOOP  
  for (Int_t ievent=0; ievent<nevH; ievent++){
    hitTree->GetEntry(ievent);
    for (int ii=0; ii<240; ii++)  { countE[ii]=0;hit_time[ii]=0; }
    for (auto& hit : *hitArray) {
      Int_t detID = hit.GetDetectorID();
      hit_time[detID] = hit.GetTime();
      hTimeHit -> Fill(detID, hit_time[detID]);
      if (hit_time[detID]<10 && detID<96)
      	std::cout<<ievent<<" "<<detID<<" time  "<<hit_time[detID]<<endl;

      countE[detID]++;
    }
    for (int ii=0; ii<208; ii++) {
      if (countE[ii]>100) {
	hMultHit -> Fill(ii, countE[ii]);
	//	std::cout<<ii<<" "<<countE[ii]<<endl;
      }
    }
  }
  TFile* fdig = TFile::Open("o2sim_digi.root");
  std::cout << " Open digits file " << std::endl;
  TTree* digTree = (TTree*)fdig->Get("o2sim");
  o2::fit::Digit * digArr = new Digit;
  digTree->SetBranchAddress("FITDigit", &digArr);
  Int_t nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  std::cout << "Found " << nevD << " events with digits " << std::endl;
  Float_t cfd[208], amp[208];
  for (Int_t iev = 0; iev < nevD; iev++) {
    digTree->GetEvent(iev);
     for (int ii=0; ii<208; ii++)  { cfd[ii]=amp[ii]=0; }
     for (const auto& d : digArr->getChDgData()) {
       Int_t mcp = d.ChId;
       cfd[mcp] = d.CFDTime;
       amp[mcp] = d.QTCAmpl;
       //      cout<<iev<<" "<<mcp<<" "<< cfd[mcp]<<" "<< amp[mcp]<<endl;
       hMultDig ->Fill(Float_t (mcp),amp[mcp]); 
       hTimeDig ->Fill(Float_t (mcp),cfd[mcp]);
     }
   }
   TFile *Hfile = new TFile("FigFit_dig_pp.root","RECREATE");
   printf("Writting histograms to root file \n");
   Hfile->cd();
  //Create a canvas, set the view range, show histograms
  //  TCanvas *c1 = new TCanvas("c1","Alice T0 Time ",400,10,600,600);
  hTimeHit->Write();
  hMultHit->Write();
  hTimeDig->Write();
  hMultDig->Write();
 

} // end of macro





