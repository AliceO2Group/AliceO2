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

#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#endif

void CheckTracks(std::string tracfile = "o2trac_its.root", std::string clusfile = "o2clus_its.root", std::string hitfile = "o2sim.root")
{
  using namespace o2::itsmft;
  using namespace o2::its;

  TFile* f = TFile::Open("CheckTracks.root", "recreate");
  TNtuple* nt = new TNtuple("ntt", "track ntuple",
                            //"mcYOut:recYOut:"
                            "mcZOut:recZOut:"
                            "mcPhiOut:recPhiOut:"
                            "mcThetaOut:recThetaOut:"
                            "mcPhi:recPhi:"
                            "mcLam:recLam:"
                            "mcPt:recPt:"
                            "ipD:ipZ:label");

  // MC tracks
  TFile* file0 = TFile::Open(hitfile.data());
  TTree* mcTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree->SetBranchAddress("MCTrack", &mcArr);
  std::vector<o2::TrackReference>* mcTrackRefs = nullptr;
  mcTree->SetBranchAddress("TrackRefs", &mcTrackRefs);

  // Clusters
  TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<Cluster>* clusArr = nullptr;
  auto* branch = clusTree->GetBranch("ITSCluster");
  if (!branch) {
    std::cout << "No clusters !" << std::endl;
    return;
  }
  branch->SetAddress(&clusArr);
  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  // Reconstructed tracks
  TFile* file1 = TFile::Open(tracfile.data());
  TTree* recTree = (TTree*)gFile->Get("o2sim");
  std::vector<TrackITS>* recArr = nullptr;
  recTree->SetBranchAddress("ITSTrack", &recArr);
  // Track MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* trkLabArr = nullptr;
  recTree->SetBranchAddress("ITSTrackMCTruth", &trkLabArr);

  Int_t nrec = 0;
  Int_t lastEventIDcl = -1, cf = 0;
  Int_t lastEventIDtr = -1, tf = 0;
  Int_t nev = mcTree->GetEntries();

  Int_t nb = 100;
  Double_t xbins[nb + 1], ptcutl = 0.01, ptcuth = 10.;
  Double_t a = TMath::Log(ptcuth / ptcutl) / nb;
  for (Int_t i = 0; i <= nb; i++)
    xbins[i] = ptcutl * TMath::Exp(i * a);
  TH1D* num = new TH1D("num", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", nb, xbins);
  num->Sumw2();
  TH1D* fak = new TH1D("fak", ";#it{p}_{T} (GeV/#it{c});Fak", nb, xbins);
  fak->Sumw2();
  TH1D* den = new TH1D("den", ";#it{p}_{T} (GeV/#it{c});Den", nb, xbins);
  den->Sumw2();

  for (Int_t n = 0; n < nev; n++) {
    std::cout << "\nMC event " << n << '/' << nev << std::endl;
    Int_t nGen = 0, nGoo = 0, nFak = 0;
    mcTree->GetEvent(n);
    Int_t nmc = mcArr->size();
    Int_t nmcrefs = mcTrackRefs->size();

    while ((n > lastEventIDtr) && (tf < recTree->GetEntries())) { // Cache a new reconstructed entry
      recTree->GetEvent(tf);
      nrec = recArr->size();
      for (int i = 0; i < nrec; i++) { // Find the last MC event within this reconstructed entry
        auto mclab = (trkLabArr->getLabels(i))[0];
        auto id = mclab.getEventID();
        if (id > lastEventIDtr)
          lastEventIDtr = id;
      }
      if (nrec > 0)
        std::cout << "Caching track entry #" << tf << ", with the last event ID=" << lastEventIDtr << std::endl;
      tf++;
    }
    while ((n > lastEventIDcl) && (cf < clusTree->GetEntries())) { // Cache a new reconstructed entry
      clusTree->GetEvent(cf);
      for (int i = 0; i < clusArr->size(); i++) { // Find the last MC event within this reconstructed entry
        auto mclab = (clusLabArr->getLabels(i))[0];
        auto id = mclab.getEventID();
        if (id > lastEventIDcl)
          lastEventIDcl = id;
      }
      if (nrec > 0)
        std::cout << "Caching cluster entry #" << cf << ", with the last event ID=" << lastEventIDcl << std::endl;
      cf++;
    }

    while (nmc--) {
      const auto& mcTrack = (*mcArr)[nmc];
      Int_t mID = mcTrack.getMotherTrackId();
      if (mID >= 0)
        continue; // Select primary particles
      Int_t pdg = mcTrack.GetPdgCode();
      if (TMath::Abs(pdg) != 211)
        continue; // Select pions

      int ok = 0;
      // Check the availability of clusters
      for (uint i = 0; i < clusArr->size(); i++) {
        const Cluster& c = (*clusArr)[i];
        auto lab = (clusLabArr->getLabels(i))[0];
        if (lab.getEventID() != n)
          continue;
        if (lab.getTrackID() != nmc)
          continue;
        auto r = c.getX();
        if (TMath::Abs(r - 2.2) < 0.5)
          ok |= 0b1;
        if (TMath::Abs(r - 3.0) < 0.5)
          ok |= 0b10;
        if (TMath::Abs(r - 3.8) < 0.5)
          ok |= 0b100;
        if (TMath::Abs(r - 19.5) < 0.5)
          ok |= 0b1000;
        if (TMath::Abs(r - 24.5) < 0.5)
          ok |= 0b10000;
        if (TMath::Abs(r - 34.5) < 0.5)
          ok |= 0b100000;
        if (TMath::Abs(r - 39.5) < 0.5)
          ok |= 0b1000000;
      }
      if (ok != 0b1111111)
        continue;

      nGen++; // Generated tracks for the efficiency calculation

      // Float_t mcYOut=-1., recYOut=-1.;
      Float_t mcZOut = -1., recZOut = -1.;
      Float_t mcPhiOut = -1., recPhiOut = -1.;
      Float_t mcThetaOut = -1., recThetaOut = -1.;
      Float_t mcPx = mcTrack.GetStartVertexMomentumX();
      Float_t mcPy = mcTrack.GetStartVertexMomentumY();
      Float_t mcPz = mcTrack.GetStartVertexMomentumZ();
      Float_t mcPhi = TMath::ATan2(mcPy, mcPx), recPhi = -1.;
      Float_t mcPt = mcTrack.GetPt(), recPt = -1.;
      Float_t mcLam = TMath::ATan2(mcPz, mcPt), recLam = -1.;
      Float_t ip[2]{0., 0.};
      Float_t label = -123456789.;

      den->Fill(mcPt);

      for (Int_t i = 0; i < nrec; i++) {
        const TrackITS& recTrack = (*recArr)[i];
        auto mclab = (trkLabArr->getLabels(i))[0];
        auto id = mclab.getEventID();
        if (id != n)
          continue;
        Int_t lab = mclab.getTrackID();
        if (TMath::Abs(lab) != nmc)
          continue;

        for (auto& ref : *mcTrackRefs) {
          if (ref.getUserId() != 6)
            continue;
          if (ref.getTrackID() != nmc)
            continue;
          // mcYOut=ref.LocalY();
          mcZOut = ref.Z();
          mcPhiOut = ref.Phi();
          mcThetaOut = ref.Theta();
          break;
        }

        auto out = recTrack.getParamOut();
        // recYOut = out.getY();
        recZOut = out.getZ();
        recPhiOut = out.getPhi();
        recThetaOut = out.getTheta();

        std::array<float, 3> p;
        recTrack.getPxPyPzGlo(p);
        recPt = recTrack.getPt();
        recPhi = TMath::ATan2(p[1], p[0]);
        recLam = TMath::ATan2(p[2], recPt);
        Float_t vx = 0., vy = 0., vz = 0.; // Assumed primary vertex
        Float_t bz = 5.;                   // Assumed magnetic field
        recTrack.getImpactParams(vx, vy, vz, bz, ip);
        label = lab;

        if (label > 0) {
          nGoo++; // Good found tracks for the efficiency calculation
          num->Fill(mcPt);
        } else {
          nFak++; // Fake-track rate calculation
          fak->Fill(mcPt);
        }
      }

      nt->Fill( // mcYOut,recYOut,
        mcZOut, recZOut, mcPhiOut, recPhiOut, mcThetaOut, recThetaOut, mcPhi, recPhi, mcLam, recLam, mcPt, recPt, ip[0],
        ip[1], label);
    }
    if (nGen > 0) {
      Float_t eff = nGoo / Float_t(nGen);
      Float_t rat = nFak / Float_t(nGen);
      std::cout << "Good found tracks: " << nGoo << ",  efficiency: " << eff << ",  fake-track rate: " << rat << std::endl;
    }
  }

  // "recPt>0" means "found tracks only"
  // "label>0" means "found good tracks only"
  new TCanvas;
  nt->Draw("ipD", "recPt>0 && label>0");
  new TCanvas;
  nt->Draw("mcLam-recLam", "recPt>0 && label>0");
  new TCanvas;
  nt->Draw("mcPt-recPt", "recPt>0 && label>0");
  new TCanvas;
  nt->Draw("mcZOut-recZOut", "recPt>0 && label>0 && abs(mcZOut-recZOut)<0.025");
  new TCanvas;
  nt->Draw("mcPhiOut-recPhiOut", "recPt>0 && label>0");
  new TCanvas;
  nt->Draw("mcThetaOut-recThetaOut", "recPt>0 && label>0");
  TCanvas* c1 = new TCanvas;
  c1->SetLogx();
  c1->SetGridx();
  c1->SetGridy();
  num->Divide(num, den, 1, 1, "b");
  num->Draw("histe");
  fak->Divide(fak, den, 1, 1, "b");
  fak->SetLineColor(2);
  fak->Draw("histesame");
  f->Write();
  f->Close();
}
