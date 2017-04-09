/// \file CheckTracks.C
/// \brief Simple macro to check ITSU tracks

#if !defined(__CINT__) || defined(__MAKECINT__)
  #include <array>

  #include <TFile.h>
  #include <TTree.h>
  #include <TClonesArray.h>
  #include <TH2F.h>
  #include <TNtuple.h>
  #include <TCanvas.h>
  #include <TMath.h>
  #include <TString.h>

  #include "SimulationDataFormat/MCTrack.h"
  #include "ITSReconstruction/Cluster.h"
  #include "ITSReconstruction/CookedTrack.h"
#endif

void CheckTracks(Int_t nEvents = 10, TString mcEngine = "TGeant3") {
  using namespace o2::ITS;

  TFile *f=TFile::Open("CheckTracks.root","recreate");
  TNtuple *nt=new TNtuple("ntt","track ntuple",
			  "mcPhi:mcLam:mcPt:recPhi:recLam:recPt:ipD:ipZ:label");

  char filename[100];

  // MC tracks
  sprintf(filename, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file0 = TFile::Open(filename);
  TTree *mcTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray mcArr("MCTrack"), *pmcArr(&mcArr);
  mcTree->SetBranchAddress("MCTrack",&pmcArr);

  // Reconstructed tracks
  sprintf(filename, "AliceO2_%s.trac_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file1 = TFile::Open(filename);
  TTree *recTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray recArr("o2::ITS::CookedTrack"), *precArr(&recArr);
  recTree->SetBranchAddress("ITSTrack",&precArr);
  
  Int_t nev=mcTree->GetEntries();
  for (Int_t n=0; n<nev; n++) {
    std::cout<<"Event "<<n<<'/'<<nev<<std::endl;
    Int_t nGen=0, nGoo=0;
    mcTree->GetEvent(n);
    Int_t nmc=mcArr.GetEntriesFast();
    recTree->GetEvent(n);
    Int_t nrec=recArr.GetEntriesFast();
    while(nmc--) {
      MCTrack *mcTrack = (MCTrack *)mcArr.UncheckedAt(nmc);
      Int_t mID = mcTrack->getMotherTrackId();
      if (mID >= 0) continue; // Select primary particles 
      Int_t pdg = mcTrack->GetPdgCode();
      if (TMath::Abs(pdg) != 211) continue;  // Select pions

      nGen++; // Generated tracks for the efficiency calculation 
      
      Double_t mcPx = mcTrack->GetStartVertexMomentumX();
      Double_t mcPy = mcTrack->GetStartVertexMomentumY();
      Double_t mcPz = mcTrack->GetStartVertexMomentumZ();
      Double_t mcPt = mcTrack->GetPt();
      Double_t mcPhi= TMath::ATan2(mcPy,mcPx);
      Double_t mcLam= TMath::ATan2(mcPz,mcPt);
      Double_t recPhi=-1.; 
      Double_t recLam=-1.; 
      Double_t recPt=-1.; 
      Double_t ip[2]{0.,0.}; 
      Double_t label=-123456789.; 
      
      for (Int_t i=0; i<nrec; i++) {
         CookedTrack *recTrack = (CookedTrack *)recArr.UncheckedAt(i);
	 Int_t lab = recTrack->getLabel();
	 if (TMath::Abs(lab) != nmc) continue;
	 std::array<float,3> p;
	 recTrack->getPxPyPz(p);
	 recPt = recTrack->getPt();
         recPhi = TMath::ATan2(p[1],p[0]);
	 recLam = TMath::ATan2(p[2],recPt);
	 Double_t vx=0., vy=0., vz=0.;  // Assumed primary vertex
	 Double_t bz=5.;                // Assumed magnetic field 
         recTrack->getImpactParams(vx, vy, vz, bz, ip);
         label = lab;
	 
	 if (label>0) nGoo++; // Good found tracks for the efficiency calculation
      }

      nt->Fill(mcPhi,mcLam,mcPt,recPhi,recLam,recPt,ip[0],ip[1],label);

    }
    Float_t eff = (nGen > 0) ? nGoo/Float_t(nGen) : -1.;
    std::cout<<"Good found tracks: "<<nGoo<<",  efficiency: "<<eff<<std::endl;
  }
  
  // "recPt>0" means "found tracks only"  
  // "label>0" means "found good tracks only"  
  new TCanvas; nt->Draw("ipD","recPt>0 && label>0");
  new TCanvas; nt->Draw("mcLam-recLam","recPt>0 && label>0");
  new TCanvas; nt->Draw("mcPt-recPt","recPt>0 && label>0");
  f->Write();
  f->Close();
}
