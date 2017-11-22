/// \file CheckTracks.C
/// \brief Simple macro to check ITSU tracks

#if !defined(__CLING__) || defined(__ROOTCLING__)
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
  #include "SimulationDataFormat/MCCompLabel.h"
  #include "SimulationDataFormat/MCTruthContainer.h"
  #include "ITSMFTReconstruction/Cluster.h"
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
  TTree *mcTree=(TTree*)gFile->Get("o2sim");
  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree->SetBranchAddress("MCTrack",&mcArr);

  // Reconstructed tracks
  sprintf(filename, "AliceO2_%s.trac_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file1 = TFile::Open(filename);
  TTree *recTree=(TTree*)gFile->Get("o2sim");
  std::vector<CookedTrack> *recArr=nullptr;
  recTree->SetBranchAddress("ITSTrack",&recArr);
  // Track MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *trkLabArr=nullptr;
  recTree->SetBranchAddress("ITSTrackMCTruth",&trkLabArr);

  Int_t tf=0, nrec=0;
  Int_t lastEventID=-1;
  Int_t nev=mcTree->GetEntries();
  for (Int_t n=0;n<nev;n++) {
    std::cout<<"\nMC event "<<n<<'/'<<nev<<std::endl;
    Int_t nGen=0, nGoo=0;
    mcTree->GetEvent(n);
    Int_t nmc=mcArr->size();

    while ((n>lastEventID) && (tf<recTree->GetEntries())) { // Cache a new reconstructed TF
       recTree->GetEvent(tf);
       nrec=recArr->size();
       for (int i=0; i<nrec; i++) { // Find the last MC event within this reconstructed TF
	 auto mclab = (trkLabArr->getLabels(i))[0];
	 auto id = mclab.getEventID();
	 if (id>lastEventID) lastEventID=id;
       }
       if (nrec>0)
	 std::cout<<"Caching TF #"<<tf<<", with the last event ID="<<lastEventID<<std::endl;
       tf++;
    }

    while(nmc--) {
      const auto& mcTrack = (*mcArr)[nmc];
      Int_t mID = mcTrack.getMotherTrackId();
      if (mID >= 0) continue;// Select primary particles 
      Int_t pdg = mcTrack.GetPdgCode();
      if (TMath::Abs(pdg) != 211) continue; // Select pions

      nGen++;// Generated tracks for the efficiency calculation 
      
      Float_t mcPx = mcTrack.GetStartVertexMomentumX();
      Float_t mcPy = mcTrack.GetStartVertexMomentumY();
      Float_t mcPz = mcTrack.GetStartVertexMomentumZ();
      Float_t mcPt = mcTrack.GetPt();
      Float_t mcPhi= TMath::ATan2(mcPy,mcPx);
      Float_t mcLam= TMath::ATan2(mcPz,mcPt);
      Float_t recPhi=-1.;
      Float_t recLam=-1.;
      Float_t recPt=-1.;
      Float_t ip[2]{0.,0.};
      Float_t label=-123456789.;
      
      for (Int_t i=0;i<nrec;i++) {
	 const CookedTrack &recTrack = (*recArr)[i];
	 auto mclab = (trkLabArr->getLabels(i))[0];
         auto id = mclab.getEventID();
         if (id != n) continue;
	 Int_t lab = mclab.getTrackID();
	 if (TMath::Abs(lab) != nmc) continue;
	 std::array<float,3> p;
	 recTrack.getPxPyPzGlo(p);
	 recPt = recTrack.getPt();
         recPhi = TMath::ATan2(p[1],p[0]);
	 recLam = TMath::ATan2(p[2],recPt);
	 Float_t vx=0., vy=0., vz=0.; // Assumed primary vertex
	 Float_t bz=5.;               // Assumed magnetic field 
         recTrack.getImpactParams(vx, vy, vz, bz, ip);
         label = lab;
	 
	 if (label>0) nGoo++;// Good found tracks for the efficiency calculation
      }

      nt->Fill(mcPhi,mcLam,mcPt,recPhi,recLam,recPt,ip[0],ip[1],label);

    }
    Float_t eff = (nGen > 0) ? nGoo/Float_t(nGen) : -1.;
    std::cout<<"Good found tracks: "<<nGoo<<",  efficiency: "<<eff<<std::endl;
  }
  
  // "recPt>0" means "found tracks only"  
  // "label>0" means "found good tracks only"  
  new TCanvas;nt->Draw("ipD","recPt>0 && label>0");
  new TCanvas;nt->Draw("mcLam-recLam","recPt>0 && label>0");
  new TCanvas;nt->Draw("mcPt-recPt","recPt>0 && label>0");
  f->Write();
  f->Close();
}
