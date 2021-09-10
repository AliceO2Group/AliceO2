/// \file CheckTracksCA.C
/// \brief Simple macro to check ITSU tracks

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TH2F.h>
#include <TCanvas.h>

#include "ITSBase/GeometryTGeo.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"

#endif

using namespace std;

struct ParticleInfo {
  int event;
  int pdg;
  float pt;
  float eta;
  float phi;
  int mother;
  int first;
  unsigned short clusters = 0u;
  unsigned char isReco = 0u;
  unsigned char isFake = 0u;
  bool isPrimary = 0u;
  unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
  o2::its::TrackITS track;
};

#pragma link C++ class ParticleInfo + ;

void CheckTracksCA(std::string tracfile = "o2trac_its.root", std::string clusfile = "o2clus_its.root", std::string kinefile = "o2sim_Kine.root")
{

  using namespace o2::itsmft;
  using namespace o2::its;

  // Geometry
  o2::base::GeometryManager::loadGeometry();
  auto gman = o2::its::GeometryTGeo::Instance();

  // MC tracks
  TFile* file0 = TFile::Open(kinefile.data());
  TTree* mcTree = (TTree*)gFile->Get("o2sim");
  mcTree->SetBranchStatus("*", 0); //disable all branches
  mcTree->SetBranchStatus("MCTrack*", 1);

  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree->SetBranchAddress("MCTrack", &mcArr);

  // Clusters
  TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  // Reconstructed tracks
  TFile* file1 = TFile::Open(tracfile.data());
  TTree* recTree = (TTree*)gFile->Get("o2sim");
  std::vector<TrackITS>* recArr = nullptr;
  recTree->SetBranchAddress("ITSTrack", &recArr);
  // Track MC labels
  std::vector<o2::MCCompLabel>* trkLabArr = nullptr;
  recTree->SetBranchAddress("ITSTrackMCTruth", &trkLabArr);

  std::cout << "** Filling particle table ... " << std::flush;
  int lastEventIDcl = -1, cf = 0;
  int nev = mcTree->GetEntriesFast();
  std::vector<std::vector<ParticleInfo>> info(nev);
  for (int n = 0; n < nev; n++) { // loop over MC events
    mcTree->GetEvent(n);
    info[n].resize(mcArr->size());
    for (unsigned int mcI{0}; mcI < mcArr->size(); ++mcI) {
      auto part = mcArr->at(mcI);
      info[n][mcI].event = n;
      info[n][mcI].pdg = part.GetPdgCode();
      info[n][mcI].pt = part.GetPt();
      info[n][mcI].phi = part.GetPhi();
      info[n][mcI].eta = part.GetEta();
      info[n][mcI].isPrimary = part.isPrimary();
    }
  }
  std::cout << "done." << std::endl;

  std::cout << "** Creating particle/clusters correspondance ... " << std::flush;
  for (int frame = 0; frame < clusTree->GetEntriesFast(); frame++) { // Cluster frames
    if (!clusTree->GetEvent(frame))
      continue;

    for (unsigned int iClus{0}; iClus < clusArr->size(); ++iClus) {
      auto lab = (clusLabArr->getLabels(iClus))[0];
      if (!lab.isValid() || lab.getSourceID() != 0 || !lab.isCorrect())
        continue;

      int trackID, evID, srcID;
      bool fake;
      lab.get(trackID, evID, srcID, fake);
      if (evID < 0 || evID >= (int)info.size()) {
        std::cout << "Cluster MC label eventID out of range" << std::endl;
        continue;
      }
      if (trackID < 0 || trackID >= (int)info[evID].size()) {
        std::cout << "Cluster MC label trackID out of range" << std::endl;
        continue;
      }

      const CompClusterExt& c = (*clusArr)[iClus];
      auto layer = gman->getLayer(c.getSensorID());
      info[evID][trackID].clusters |= 1 << layer;
    }
  }
  std::cout << "done." << std::endl;

  std::cout << "** Analysing tracks ... " << std::flush;
  int unaccounted{0}, good{0}, fakes{0}, total{0};
  for (int frame = 0; frame < recTree->GetEntriesFast(); frame++) { // Cluster frames
    if (!recTree->GetEvent(frame))
      continue;
    total += trkLabArr->size();
    for (unsigned int iTrack{0}; iTrack < trkLabArr->size(); ++iTrack) {
      auto lab = trkLabArr->at(iTrack);
      if (!lab.isSet()) {
        unaccounted++;
        continue;
      }
      int trackID, evID, srcID;
      bool fake;
      lab.get(trackID, evID, srcID, fake);
      if (evID < 0 || evID >= (int)info.size()) {
        unaccounted++;
        continue;
      }
      if (trackID < 0 || trackID >= (int)info[evID].size()) {
        unaccounted++;
        continue;
      }
      info[evID][trackID].isReco += !fake;
      info[evID][trackID].isFake += fake;
      /// We keep the best track we would keep in the data
      if (recArr->at(iTrack).isBetter(info[evID][trackID].track, 1.e9)) {
        info[evID][trackID].storedStatus = fake;
        info[evID][trackID].track = recArr->at(iTrack);
      }

      fakes += fake;
      good += !fake;
    }
  }
  std::cout << "done." << std::endl;

  std::cout << "** Some statistics:" << std::endl;
  std::cout << "\t- Total number of tracks: " << total << std::endl;
  std::cout << "\t- Total number of tracks not corresponding to particles: " << unaccounted << " (" << unaccounted * 100. / total << "%)" << std::endl;
  std::cout << "\t- Total number of fakes: " << fakes << " (" << fakes * 100. / total << "%)" << std::endl;
  std::cout << "\t- Total number of good: " << good << " (" << good * 100. / total << "%)" << std::endl;

  int nb = 100;
  double xbins[nb + 1], ptcutl = 0.01, ptcuth = 10.;
  double a = std::log(ptcuth / ptcutl) / nb;
  for (int i = 0; i <= nb; i++)
    xbins[i] = ptcutl * std::exp(i * a);
  TH1D* num = new TH1D("num", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", nb, xbins);
  num->Sumw2();
  TH1D* numChi2 = new TH1D("numChi2", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", 200, 0, 100);

  TH1D* fak = new TH1D("fak", ";#it{p}_{T} (GeV/#it{c});Fak", nb, xbins);
  fak->Sumw2();
  TH1D* multiFak = new TH1D("multiFak", ";#it{p}_{T} (GeV/#it{c});Fak", nb, xbins);
  multiFak->Sumw2();
  TH1D* fakChi2 = new TH1D("fakChi2", ";#it{p}_{T} (GeV/#it{c});Fak", 200, 0, 100);

  TH1D* clone = new TH1D("clone", ";#it{p}_{T} (GeV/#it{c});Clone", nb, xbins);
  clone->Sumw2();

  TH1D* den = new TH1D("den", ";#it{p}_{T} (GeV/#it{c});Den", nb, xbins);
  den->Sumw2();

  for (auto& evInfo : info) {
    for (auto& part : evInfo) {
      if (part.clusters != 0x7f) {
        // part.clusters != 0x3f && part.clusters != 0x3f << 1 &&
        // part.clusters != 0x1f && part.clusters != 0x1f << 1 && part.clusters != 0x1f << 2 &&
        // part.clusters != 0x0f && part.clusters != 0x0f << 1 && part.clusters != 0x0f << 2 && part.clusters != 0x0f << 3) {
        continue;
      }
      if (std::abs(part.eta) > 1.1 && !part.isPrimary) {
        continue;
      }
      den->Fill(part.pt);
      if (part.isReco) {
        num->Fill(part.pt);
        if (part.isReco > 1) {
          for (int _i{0}; _i < part.isReco - 1; ++_i) {
            clone->Fill(part.pt);
          }
        }
      }
      if (part.isFake) {
        fak->Fill(part.pt);
        if (part.isFake > 1) {
          for (int _i{0}; _i < part.isFake - 1; ++_i) {
            multiFak->Fill(part.pt);
          }
        }
      }
    }
  }

  TCanvas* c1 = new TCanvas;
  c1->SetLogx();
  c1->SetGridx();
  c1->SetGridy();
  TH1* sum = (TH1*)num->Clone("sum");
  sum->Add(fak);
  sum->Divide(sum, den, 1, 1);
  sum->SetLineColor(kBlack);
  sum->Draw("hist");
  num->Divide(num, den, 1, 1, "b");
  num->Draw("histesame");
  fak->Divide(fak, den, 1, 1, "b");
  fak->SetLineColor(2);
  fak->Draw("histesame");
  multiFak->Divide(multiFak, den, 1, 1, "b");
  multiFak->SetLineColor(kRed + 1);
  multiFak->Draw("histsame");
  clone->Divide(clone, den, 1, 1, "b");
  clone->SetLineColor(3);
  clone->Draw("histesame");

  std::cout << "** Streaming output TTree to file ... " << std::flush;
  TFile file("CheckTracksCA.root", "recreate");
  TTree tree("ParticleInfo", "ParticleInfo");
  ParticleInfo pInfo;
  tree.Branch("particle", &pInfo);
  for (auto& event : info) {
    for (auto& part : event) {
      int nCl{0};
      for (unsigned int bit{0}; bit < sizeof(pInfo.clusters) * 8; ++bit) {
        nCl += bool(part.clusters & (1 << bit));
      }
      if (nCl < 3) {
        continue;
      }
      pInfo = part;
      tree.Fill();
    }
  }
  tree.Write();
  sum->Write("total");
  fak->Write("singleFake");
  num->Write("efficiency");
  multiFak->Write("multiFake");
  clone->Write("clones");
  file.Close();
  std::cout << " done." << std::endl;
}
