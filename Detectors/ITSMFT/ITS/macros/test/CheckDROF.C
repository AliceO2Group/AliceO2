// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include "TH1F.h"
#include <TH2F.h>
#include "TH2D.h"
#include "TH3D.h"
#include <TProfile.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPad.h>
#include <TRatioPlot.h>

#include "ITSBase/GeometryTGeo.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "DetectorsBase/Propagator.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITS/Vertex.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/DigitizationContext.h"

#endif

using namespace std;

void plotHistos(TFile* fWO, TFile* f, const char* append = "");

struct ParticleInfo { // particle level information for tracks
  int event;
  int pdg;
  float pt;
  float eta;
  float phi;
  int mother;
  int first;
  float pvx{};
  float pvy{};
  float pvz{};
  float dcaxy;
  float dcaz;
  unsigned short clusters = 0u;
  unsigned char isReco = 0u;
  unsigned char isFake = 0u;
  bool isPrimary = false;
  int bcInROF{-1};
  int rofId{-1};
  unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
  o2::its::TrackITS track;

  void print() const
  {
    LOGP(info, "event={} pdg={} pt={} eta={} phi={} mother={} clusters={:7b} isReco={} isFake={} isPrimary={} bcInROF={} rofId={} | {}", event, pdg, pt, eta, phi, mother, clusters, isReco, isFake, isPrimary, bcInROF, rofId, track.asString());
  }

  int getNClusters() const noexcept
  {
    int nCl{0};
    for (unsigned int bit{0}; bit < sizeof(ParticleInfo::clusters) * 8; ++bit) {
      nCl += bool(clusters & (1 << bit));
    }
    return nCl;
  }

  bool isReconstructable() const noexcept
  {
    return isPrimary && (7 == getNClusters()) && bcInROF >= 0;
  }
};
#pragma link C++ class ParticleInfo + ;
#pragma link C++ class std::vector < ParticleInfo> + ;

struct VertexInfo {       // Vertex level info
  float purity;           // fraction of main cont. labels to all
  o2::its::Vertex vertex; // reconstructed vertex
  int bcInROF{-1};
  int rofId{-1};
  int event{-1};                       // corresponding MC event
  std::vector<o2::MCCompLabel> labels; // contributor labels
  o2::MCCompLabel mainLabel;           // main label

  void computeMain()
  {
    std::unordered_map<o2::MCCompLabel, size_t> freq;
    size_t totalSet = 0;

    // Count frequencies of set labels
    for (auto const& lab : labels) {
      if (lab.isSet()) {
        ++freq[lab];
        ++totalSet;
      }
    }
    if (totalSet == 0) {
      return;
    }
    // Find the label with maximum count
    auto best = std::max_element(freq.begin(), freq.end(), [](auto const& a, auto const& b) { return a.second < b.second; });
    size_t maxCount = best->second;

    // If there's no majority (all counts == 1), fall back to first set label
    o2::MCCompLabel mainLab;
    if (maxCount == 1) {
      for (auto const& lab : labels) {
        if (lab.isSet()) {
          mainLab = lab;
          break;
        }
      }
    } else {
      mainLab = best->first;
    }
    purity = (float)maxCount / (float)labels.size();
  }
};
#pragma link C++ class VertexInfo + ;

using namespace o2::itsmft;
using namespace o2::its;

void CheckDROF(bool plot = false, bool write = false, const std::string& tracfile = "o2trac_its.root",
               const std::string& magfile = "o2sim_grp.root",
               const std::string& clusfile = "o2clus_its.root",
               const std::string& kinefile = "o2sim_Kine.root")
{
  constexpr int64_t roFrameLengthInBC = 198; // for pp=198
  constexpr int64_t roFrameBiasInBC = 64;    // ITS delay accounted for in digitization
  constexpr float roFbins{roFrameLengthInBC + 2.f};
  constexpr int bcValStart{60}, bcValEnd{140}; // adjustable region of validation train

  if (!plot) {
    int trackID, evID, srcID;
    bool fake;

    // Magnetic field and Propagator
    o2::base::Propagator::initFieldFromGRP(magfile);
    float bz = o2::base::Propagator::Instance()->getNominalBz();

    // Geometry
    o2::base::GeometryManager::loadGeometry();
    auto gman = o2::its::GeometryTGeo::Instance();

    // MC tracks
    TFile* file0 = TFile::Open(kinefile.data());
    TTree* mcTree = (TTree*)gFile->Get("o2sim");
    mcTree->SetBranchStatus("*", 0); // disable all branches
    mcTree->SetBranchStatus("MCTrack*", 1);
    mcTree->SetBranchStatus("MCEventHeader*", 1);
    std::vector<o2::MCTrack>* mcArr = nullptr;
    mcTree->SetBranchAddress("MCTrack", &mcArr);
    o2::dataformats::MCEventHeader* mcEvent = nullptr;
    mcTree->SetBranchAddress("MCEventHeader.", &mcEvent);

    auto* dc = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
    const auto& irs = dc->getEventRecords();
    dc->printCollisionSummary(false, 20);

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
    std::vector<o2::itsmft::ROFRecord> rofRecVec, *rofRecVecP = &rofRecVec;
    recTree->SetBranchAddress("ITSTracksROF", &rofRecVecP);
    // Vertices
    std::vector<o2::its::Vertex>* recVerArr = nullptr;
    recTree->SetBranchAddress("Vertices", &recVerArr);
    std::vector<ROFRecord>* recVerROFArr = nullptr;
    recTree->SetBranchAddress("VerticesROF", &recVerROFArr);
    std::vector<o2::MCCompLabel>* recVerLabelsArr = nullptr;
    recTree->SetBranchAddress("ITSVertexMCTruth", &recVerLabelsArr);
    std::vector<float>* recVerPurityArr = nullptr;
    recTree->SetBranchAddress("ITSVertexMCPurity", &recVerPurityArr);

    std::cout << "** Filling particle table ... " << std::flush;
    int lastEventIDcl = -1, cf = 0;
    const int nev = mcTree->GetEntriesFast();
    std::vector<std::vector<ParticleInfo>> info;
    info.resize(nev);
    TH1D* hZvertex = new TH1D("hZvertex", "Z vertex", 100, -20, 20);
    for (int n = 0; n < nev; n++) { // loop over MC events
      mcTree->GetEvent(n);
      info[n].resize(mcArr->size());
      hZvertex->Fill(mcEvent->GetZ());
      const auto& ir = irs[mcEvent->GetEventID() - 1]; // event id start from 1
      for (unsigned int mcI{0}; mcI < mcArr->size(); ++mcI) {
        auto part = mcArr->at(mcI);
        info[n][mcI].event = n;
        info[n][mcI].pdg = part.GetPdgCode();
        info[n][mcI].pvx = mcEvent->GetX();
        info[n][mcI].pvy = mcEvent->GetY();
        info[n][mcI].pvz = mcEvent->GetZ();
        info[n][mcI].pt = part.GetPt();
        info[n][mcI].phi = part.GetPhi();
        info[n][mcI].eta = part.GetEta();
        info[n][mcI].isPrimary = part.isPrimary();
        if (!ir.isDummy()) {
          info[n][mcI].bcInROF = (ir.toLong() - roFrameBiasInBC) % roFrameLengthInBC;
          info[n][mcI].rofId = (ir.toLong() - roFrameBiasInBC) / roFrameLengthInBC;
        }
      }
    }
    std::cout << "done." << std::endl;

    std::cout << "** Creating particle/clusters correspondance ... " << std::flush;
    for (int frame = 0; frame < clusTree->GetEntriesFast(); frame++) { // Cluster frames
      if (!clusTree->GetEvent(frame)) {
        continue;
      }

      for (unsigned int iClus{0}; iClus < clusArr->size(); ++iClus) {
        auto lab = (clusLabArr->getLabels(iClus))[0];
        if (!lab.isValid() || lab.getSourceID() != 0 || !lab.isCorrect()) {
          continue;
        }

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

    std::cout << "** Analysing tracks... " << std::flush;
    int unaccounted{0}, good{0}, fakes{0}, total{0}, length{0};
    for (int frame = 0; frame < recTree->GetEntriesFast(); frame++) { // Cluster frames
      if (!recTree->GetEvent(frame)) {
        continue;
      }
      total += trkLabArr->size();
      for (unsigned int iTrack{0}; iTrack < trkLabArr->size(); ++iTrack) {
        auto lab = trkLabArr->at(iTrack);
        if (!lab.isSet()) {
          unaccounted++;
          continue;
        }
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
          info[evID][trackID].track = recArr->at(iTrack);
          info[evID][trackID].storedStatus = fake;
          static float ip[2]{0., 0.};
          info[evID][trackID].track.getImpactParams(info[evID][trackID].pvx, info[evID][trackID].pvy, info[evID][trackID].pvz, bz, ip);
          info[evID][trackID].dcaxy = ip[0];
          info[evID][trackID].dcaz = ip[1];
        }

        fakes += fake;
        good += !fake;
        if (!fake) {
          for (unsigned int bit{0}; bit < 7; ++bit) {
            length += bool(info[evID][trackID].clusters & (1 << bit));
          }
        }
      }
    }
    std::cout << "done." << std::endl;
    std::cout << "** Some statistics:" << std::endl;
    std::cout << "\t- Total number of tracks: " << total << std::endl;
    std::cout << "\t- Total number of tracks not corresponding to particles: " << unaccounted << " (" << unaccounted * 100. / total << "%)" << std::endl;
    std::cout << "\t- Total number of fakes: " << fakes << " (" << fakes * 100. / total << "%)" << std::endl;
    std::cout << "\t- Total number of good: " << good << " (" << good * 100. / total << "%)" << std::endl;
    std::cout << "\t- Average length of good tracks: " << (double)length / (double)good << std::endl;

    TFile* fOut{nullptr};
    if (write) {
      fOut = TFile::Open("checkDROF.root", "RECREATE");
    }

    const int nb = 100;
    double xbins[nb + 1], ptcutl = 0.01, ptcuth = 10.;
    double a = std::log(ptcuth / ptcutl) / nb;
    for (int i = 0; i <= nb; i++) {
      xbins[i] = ptcutl * std::exp(i * a);
    }

    //////////////////////
    // Eff Tracks
    {
      auto num = new TH2D("num", ";#it{p}_{T} (GeV/#it{c});NCls;Efficiency (fake-track rate)", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      num->Sumw2();
      auto fak = new TH2D("fak", ";#it{p}_{T} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      fak->Sumw2();
      auto multiFak = new TH2D("multiFak", ";#it{p}_{T} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      multiFak->Sumw2();
      auto clone = new TH2D("clone", ";#it{p}_{T} (GeV/#it{c});NCls;Clone", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      clone->Sumw2();
      auto den = new TH2D("den", ";#it{p}_{T} (GeV/#it{c});NCls;Den", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      den->Sumw2();
      auto numMC = new TH2D("numMC", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Efficiency (fake-track rate)", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      numMC->Sumw2();
      auto fakMC = new TH2D("fakMC", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      fakMC->Sumw2();
      auto multiFakMC = new TH2D("multiFakMC", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      multiFakMC->Sumw2();
      auto cloneMC = new TH2D("cloneMC", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Clone", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      cloneMC->Sumw2();
      auto denMC = new TH2D("denMC", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Den", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      denMC->Sumw2();

      auto numVal = new TH2D("numVal", ";#it{p}_{T} (GeV/#it{c});NCls;Efficiency (fake-track rate)", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      numVal->Sumw2();
      auto fakVal = new TH2D("fakVal", ";#it{p}_{T} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      fakVal->Sumw2();
      auto multiFakVal = new TH2D("multiFakVal", ";#it{p}_{T} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      multiFakVal->Sumw2();
      auto cloneVal = new TH2D("cloneVal", ";#it{p}_{T} (GeV/#it{c});NCls;Clone", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      cloneVal->Sumw2();
      auto denVal = new TH2D("denVal", ";#it{p}_{T} (GeV/#it{c});NCls;Den", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      denVal->Sumw2();
      auto numMCVal = new TH2D("numMCVal", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Efficiency (fake-track rate)", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      numMCVal->Sumw2();
      auto fakMCVal = new TH2D("fakMCVal", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      fakMCVal->Sumw2();
      auto multiFakMCVal = new TH2D("multiFakMCVal", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      multiFakMCVal->Sumw2();
      auto cloneMCVal = new TH2D("cloneMCVal", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Clone", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      cloneMCVal->Sumw2();
      auto denMCVal = new TH2D("denMCVal", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Den", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      denMCVal->Sumw2();

      auto numMig = new TH2D("numMig", ";#it{p}_{T} (GeV/#it{c});NCls;Efficiency (fake-track rate)", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      numMig->Sumw2();
      auto fakMig = new TH2D("fakMig", ";#it{p}_{T} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      fakMig->Sumw2();
      auto multiFakMig = new TH2D("multiFakMig", ";#it{p}_{T} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      multiFakMig->Sumw2();
      auto cloneMig = new TH2D("cloneMig", ";#it{p}_{T} (GeV/#it{c});NCls;Clone", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      cloneMig->Sumw2();
      auto denMig = new TH2D("denMig", ";#it{p}_{T} (GeV/#it{c});NCls;Den", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      denMig->Sumw2();
      auto numMCMig = new TH2D("numMCMig", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Efficiency (fake-track rate)", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      numMCMig->Sumw2();
      auto fakMCMig = new TH2D("fakMCMig", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      fakMCMig->Sumw2();
      auto multiFakMCMig = new TH2D("multiFakMCMig", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Fak", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      multiFakMCMig->Sumw2();
      auto cloneMCMig = new TH2D("cloneMCMig", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Clone", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      cloneMCMig->Sumw2();
      auto denMCMig = new TH2D("denMCMig", ";#it{p}_{T,MC} (GeV/#it{c});NCls;Den", nb, xbins, 4, 4 - 0.5, 8 - 0.5);
      denMCMig->Sumw2();

      TProfile* avgClsZ = new TProfile("avgClsZ", "good attachment;z_{MC};<Cls>", 25, -20, 20);
      avgClsZ->SetLineColor(kBlack);
      TProfile* avgClsZGood = new TProfile("avgClsZGood", "good attachment;z_{MC};<Cls>", 25, -20, 20);
      avgClsZGood->SetLineColor(kBlue);
      TProfile* avgClsZFake = new TProfile("avgClsZFake", "fake attachment;z_{MC};<Cls>", 25, -20, 20);
      avgClsZFake->SetLineColor(kRed);

      for (auto& evInfo : info) {
        for (auto& part : evInfo) {
          if (!part.isReconstructable()) {
            continue;
          }
          den->Fill(part.track.getPt(), part.track.getNClusters());
          denMC->Fill(part.pt, part.track.getNClusters());
          if (part.isReco) {
            num->Fill(part.track.getPt(), part.track.getNClusters());
            numMC->Fill(part.pt, part.track.getNClusters());
            if (part.isReco > 1) {
              for (int _i{0}; _i < part.isReco - 1; ++_i) {
                clone->Fill(part.track.getPt(), part.track.getNClusters());
                cloneMC->Fill(part.pt, part.track.getNClusters());
              }
            }
          }
          if (part.isFake) {
            fak->Fill(part.track.getPt(), part.track.getNClusters());
            fakMC->Fill(part.pt, part.track.getNClusters());
            if (part.isFake > 1) {
              for (int _i{0}; _i < part.isFake - 1; ++_i) {
                multiFak->Fill(part.track.getPt(), part.track.getNClusters());
                multiFakMC->Fill(part.pt, part.track.getNClusters());
              }
            }
          }

          // sep into validation and migration region
          if (bcValStart < part.bcInROF && part.bcInROF < bcValEnd) {
            denVal->Fill(part.track.getPt(), part.track.getNClusters());
            denMCVal->Fill(part.pt, part.track.getNClusters());
            if (part.isReco) {
              numVal->Fill(part.track.getPt(), part.track.getNClusters());
              numMCVal->Fill(part.pt, part.track.getNClusters());
              if (part.isReco > 1) {
                for (int _i{0}; _i < part.isReco - 1; ++_i) {
                  cloneVal->Fill(part.track.getPt(), part.track.getNClusters());
                  cloneMCVal->Fill(part.pt, part.track.getNClusters());
                }
              }
            }
            if (part.isFake) {
              fakVal->Fill(part.track.getPt(), part.track.getNClusters());
              fakMCVal->Fill(part.pt, part.track.getNClusters());
              if (part.isFake > 1) {
                for (int _i{0}; _i < part.isFake - 1; ++_i) {
                  multiFakVal->Fill(part.track.getPt(), part.track.getNClusters());
                  multiFakMCVal->Fill(part.pt, part.track.getNClusters());
                }
              }
            }
          } else {
            denMig->Fill(part.track.getPt(), part.track.getNClusters());
            denMCMig->Fill(part.pt, part.track.getNClusters());
            if (part.isReco) {
              numMig->Fill(part.track.getPt(), part.track.getNClusters());
              numMCMig->Fill(part.pt, part.track.getNClusters());
              if (part.isReco > 1) {
                for (int _i{0}; _i < part.isReco - 1; ++_i) {
                  cloneMig->Fill(part.track.getPt(), part.track.getNClusters());
                  cloneMCMig->Fill(part.pt, part.track.getNClusters());
                }
              }
            }
            if (part.isFake) {
              fakMig->Fill(part.track.getPt(), part.track.getNClusters());
              fakMCMig->Fill(part.pt, part.track.getNClusters());
              if (part.isFake > 1) {
                for (int _i{0}; _i < part.isFake - 1; ++_i) {
                  multiFakMig->Fill(part.track.getPt(), part.track.getNClusters());
                  multiFakMCMig->Fill(part.pt, part.track.getNClusters());
                }
              }
            }
          }

          int nCl = part.getNClusters();
          avgClsZ->Fill(part.pvz, nCl);
          if (part.isReco) {
            avgClsZGood->Fill(part.pvz, nCl);
          }
          if (part.isFake) {
            avgClsZFake->Fill(part.pvz, nCl);
          }
        }
      }

      auto sum = (TH2D*)num->Clone("sum");
      auto sumMC = (TH2D*)numMC->Clone("sumMC");
      sum->Add(fak);
      sumMC->Add(fakMC);
      sum->SetLineColor(kBlack);
      sumMC->SetLineColor(kBlack);
      fak->SetLineColor(2);
      fakMC->SetLineColor(2);
      multiFak->SetLineColor(kRed + 1);
      multiFakMC->SetLineColor(kRed + 1);

      auto sumVal = (TH2D*)numVal->Clone("sumVal");
      auto sumMCVal = (TH2D*)numMCVal->Clone("sumMCVal");
      sumVal->Add(fakVal);
      sumMCVal->Add(fakMCVal);
      sumVal->SetLineColor(kBlack);
      sumMCVal->SetLineColor(kBlack);
      fakVal->SetLineColor(2);
      fakMCVal->SetLineColor(2);
      multiFakVal->SetLineColor(kRed + 1);
      multiFakMCVal->SetLineColor(kRed + 1);

      auto sumMig = (TH2D*)numMig->Clone("sumMig");
      auto sumMCMig = (TH2D*)numMCMig->Clone("sumMCMig");
      sumMig->Add(fakMig);
      sumMCMig->Add(fakMCMig);
      sumMig->SetLineColor(kBlack);
      sumMCMig->SetLineColor(kBlack);
      fakMig->SetLineColor(2);
      fakMCMig->SetLineColor(2);
      multiFakMig->SetLineColor(kRed + 1);
      multiFakMCMig->SetLineColor(kRed + 1);

      if (write) {
        num->Write();
        den->Write();
        sum->Write();
        fak->Write();
        multiFak->Write();
        numMC->Write();
        denMC->Write();
        sumMC->Write();
        fakMC->Write();
        multiFakMC->Write();

        numVal->Write();
        denVal->Write();
        sumVal->Write();
        fakVal->Write();
        multiFakVal->Write();
        numMCVal->Write();
        denMCVal->Write();
        sumMCVal->Write();
        fakMCVal->Write();
        multiFakMCVal->Write();

        numMig->Write();
        denMig->Write();
        sumMig->Write();
        fakMig->Write();
        multiFakMig->Write();
        numMCMig->Write();
        denMCMig->Write();
        sumMCMig->Write();
        fakMCMig->Write();
        multiFakMCMig->Write();
      } else {
        TCanvas* c1 = new TCanvas;
        c1->SetLogx();
        c1->SetGrid();
        gPad->DrawFrame(ptcutl, 0.05, ptcuth, 1.03, ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)");

        auto denp = den->ProjectionX();
        auto nump = num->ProjectionX();
        auto fakp = fak->ProjectionX();
        auto multiFakp = multiFak->ProjectionX();
        auto sump = sum->ProjectionX();
        auto clonep = clone->ProjectionX();

        sump->Divide(sump, denp, 1, 1, "B");
        sump->Draw("hist;same");
        nump->Divide(nump, denp, 1, 1, "B");
        nump->Draw("hist;same");
        fakp->Divide(fakp, denp, 1, 1, "B");
        fakp->Draw("hist;same");
        multiFakp->Divide(multiFakp, denp, 1, 1, "B");
        multiFakp->Draw("hist;same");
        clonep->Divide(clonep, denp, 1, 1, "B");
        clonep->SetLineColor(3);
        clonep->Draw("hist;same");

        TCanvas* c2 = new TCanvas;
        c2->Divide(2, 1);
        c2->cd(1);
        hZvertex->Draw();
        c2->cd(2);
        avgClsZ->Draw();
        avgClsZGood->Draw("same");
        avgClsZFake->Draw("same");
      }
    }

    //////////////////////
    // DROF Tracks
    {
      auto hBC = new TH1F("hBC", "Distance in BC;bcInROF;counts.", roFbins, -0.5, roFbins - 0.5);
      auto hBCTracksDen = new TH2F("hBCTracksDen", "BC Den Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);
      auto hBCTracksNum = new TH2F("hBCTracksNum", "BC Num Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);
      auto hBCTracksFake = new TH2F("hBCTracksFake", "BC Fake Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);
      auto hBCTracksSum = new TH2F("hBCTracksSum", "BC Sum Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);

      // control region
      auto hBCTracksDenVal = new TH2F("hBCTracksDenVal", "Val BC Den Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);
      auto hBCTracksNumVal = new TH2F("hBCTracksNumVal", "Val BC Num Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);
      auto hBCTracksFakeVal = new TH2F("hBCTracksFakeVal", "Val BC Fake Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);
      auto hBCTracksSumVal = new TH2F("hBCTracksSumVal", "Val BC Sum Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);

      // migration region
      auto hBCTracksDenMig = new TH2F("hBCTracksDenMig", "MigBC Den Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);
      auto hBCTracksNumMig = new TH2F("hBCTracksNumMig", "MigBC Num Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);
      auto hBCTracksFakeMig = new TH2F("hBCTracksFakeMig", "MigBC Fake Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);
      auto hBCTracksSumMig = new TH2F("hBCTracksSumMig", "MigBC Sum Tracks;bcInROF;NCls;eff.", roFbins, -0.5, roFbins - 0.5, 4, 4 - 0.5, 8 - 0.5);

      for (auto& evInfo : info) {
        for (auto& part : evInfo) {
          if (!part.isReconstructable()) {
            continue;
          }
          hBC->Fill(part.bcInROF);
          hBCTracksDen->Fill(part.bcInROF, part.track.getNClusters());
          if (part.isReco) {
            hBCTracksNum->Fill(part.bcInROF, part.track.getNClusters());
          }
          if (part.isFake) {
            hBCTracksFake->Fill(part.bcInROF, part.track.getNClusters());
          }

          if (bcValStart < part.bcInROF && part.bcInROF < bcValEnd) {
            hBCTracksDenVal->Fill(part.bcInROF, part.track.getNClusters());
            if (part.isReco) {
              hBCTracksNumVal->Fill(part.bcInROF, part.track.getNClusters());
            }
            if (part.isFake) {
              hBCTracksFakeVal->Fill(part.bcInROF, part.track.getNClusters());
            }
          } else {
            hBCTracksDenMig->Fill(part.bcInROF, part.track.getNClusters());
            if (part.isReco) {
              hBCTracksNumMig->Fill(part.bcInROF, part.track.getNClusters());
            }
            if (part.isFake) {
              hBCTracksFakeMig->Fill(part.bcInROF, part.track.getNClusters());
            }
          }
        }
      }

      hBCTracksSum->Add(hBCTracksNum);
      hBCTracksSum->Add(hBCTracksFake);
      hBCTracksSum->SetLineColor(kBlack);
      hBCTracksFake->SetLineColor(2);

      hBCTracksSumVal->Add(hBCTracksNum);
      hBCTracksSumVal->Add(hBCTracksFake);
      hBCTracksSumVal->SetLineColor(kBlack);
      hBCTracksFakeVal->SetLineColor(2);

      hBCTracksSumMig->Add(hBCTracksNum);
      hBCTracksSumMig->Add(hBCTracksFake);
      hBCTracksSumMig->SetLineColor(kBlack);
      hBCTracksFakeMig->SetLineColor(2);

      if (write) {
        hBCTracksDen->Write();
        hBCTracksNum->Write();
        hBCTracksFake->Write();
        hBCTracksSum->Write();

        hBCTracksDenVal->Write();
        hBCTracksNumVal->Write();
        hBCTracksFakeVal->Write();
        hBCTracksSumVal->Write();

        hBCTracksDenMig->Write();
        hBCTracksNumMig->Write();
        hBCTracksFakeMig->Write();
        hBCTracksSumMig->Write();
      } else {
        auto hBCTracksDenp = hBCTracksDen->ProjectionX();
        auto hBCTracksSump = hBCTracksSum->ProjectionX();
        auto hBCTracksNump = hBCTracksNum->ProjectionX();
        auto hBCTracksFakep = hBCTracksFake->ProjectionX();

        hBCTracksSump->Divide(hBCTracksSump, hBCTracksDenp, 1., 1., "B");
        hBCTracksNump->Divide(hBCTracksNump, hBCTracksDenp, 1., 1., "B");
        hBCTracksFakep->Divide(hBCTracksFakep, hBCTracksDenp, 1., 1., "B");

        auto c = new TCanvas;
        c->Divide(2, 1);
        c->cd(1);
        hBC->Draw();
        c->cd(2);
        gPad->DrawFrame(-0.5, 1e-3, roFbins - 0.5, 1.1, "Tracking >4 ITS cls;bcInROF;eff.");
        gPad->SetGrid();
        hBCTracksSump->Draw("histe;same");
        hBCTracksNump->Draw("histe;same");
        hBCTracksFakep->Draw("histe;same");
        auto leg = new TLegend;
        leg->AddEntry(hBCTracksSump, "Sum");
        leg->AddEntry(hBCTracksNump, "Good");
        leg->AddEntry(hBCTracksFakep, "Fake");
        leg->Draw();
      }
    }

    //////////////////////
    // DROF Vertices
    if constexpr (false) {
      std::vector<VertexInfo> vertexInfo;
      std::cout << "** Creating vertices/particles correspondance ... " << std::flush;
      for (int frame = 0; frame < recTree->GetEntriesFast(); frame++) { // Vertices frames
        if (!recTree->GetEvent(frame)) {
          continue;
        }
        int contLabIdx{0}; // contributor labels are stored as flat vector
        for (size_t iRecord{0}; iRecord < recVerROFArr->size(); ++iRecord) {
          auto& rec = recVerROFArr->at(iRecord);
          auto verStartIdx = rec.getFirstEntry(), verSize = rec.getNEntries();
          for (int iVertex{rec.getFirstEntry()}; iVertex < verStartIdx + verSize; ++iVertex) {
            auto& info = vertexInfo.emplace_back();
            info.vertex = recVerArr->at(iVertex);
            info.mainLabel = recVerLabelsArr->at(contLabIdx);
            info.purity = recVerPurityArr->at(contLabIdx);
            info.event = info.mainLabel.getEventID();
            ++contLabIdx;
            if (info.mainLabel.isSet()) {
              const auto& ir = irs[info.event];
              // LOGP(info, "iROF={} {} to {}", iRecord, info.mainLabel.asString(), ir.asString());
              if (!ir.isDummy()) {
                info.bcInROF = (ir.toLong() - roFrameBiasInBC) % roFrameLengthInBC;
                info.rofId = (ir.toLong() - roFrameBiasInBC) / roFrameLengthInBC;
              }
            }
          }
        }
      }
      std::cout << "done." << std::endl;

      auto hMCVtxZ = new TH1F("hMCVtxZ", "MC Vertex;Z", 50, -16, 16);
      auto hReVtxZ = new TH1F("hRecoVtxZ", "Reco Vertex;Z", 50, -16, 16);

      auto hBCVtxDen = new TH1F("hBCVtxDen;bcInROF;eff.", "BC Den Vertices", roFbins, -0.5, roFbins - 0.5);
      auto hBCVtxNum = new TH1F("hBCVtxNum;bcInROF;eff.", "BC Num Vertices", roFbins, -0.5, roFbins - 0.5);

      auto hBCVtxZDen = new TH2F("hBCVtxDen;bcInROF;z;eff.", "BC Den Vertices vs. z position", roFbins, -0.5, roFbins - 0.5, 40, -20, 20);
      auto hBCVtxZNum = new TH2F("hBCVtxNum;bcInROF;z;eff.", "BC Num Vertices vs. z position", roFbins, -0.5, roFbins - 0.5, 40, -20, 20);

      auto pBCPurity = new TProfile("pBCProfile", ";bcInROF;<purity>", roFbins, -0.5, roFbins - 0.5);
      auto pBCPurityDup = new TProfile("pBCProfileDup", ";bcInROF;<purity>", roFbins, -0.5, roFbins - 0.5);
      pBCPurityDup->SetLineColor(kRed);

      auto hVtxMCx = new TH2F("hVtxMCx", ";MC_{x};Vtx_{x}", 100, -0.3, 0.3, 100, -0.3, 0.3);
      auto hVtxMCy = new TH2F("hVtxMCy", ";MC_{y};Vtx_{y}", 100, -0.3, 0.3, 100, -0.3, 0.3);
      auto hVtxMCz = new TH2F("hVtxMCz", ";MC_{z};Vtx_{z}", 100, -20, 20, 100, -20, 20);

      for (int n = 0; n < nev; n++) { // loop over MC events
        mcTree->GetEvent(n);
        hMCVtxZ->Fill(mcEvent->GetZ());
        const auto& ir = irs[mcEvent->GetEventID() - 1]; // event id start from 1
        if (!ir.isDummy()) {
          int bcInROF = (ir.toLong() - roFrameBiasInBC) % roFrameLengthInBC;
          hBCVtxDen->Fill(bcInROF);
          hBCVtxZDen->Fill(bcInROF, mcEvent->GetZ());
        }
      }
      std::unordered_map<o2::MCCompLabel, size_t> seenMCEvent;
      for (const auto& vtx : vertexInfo) {
        ++seenMCEvent[vtx.mainLabel];
      }
      // for (const auto& [k, f] : seenMCEvent) {
      //   LOGP(info, "{}:{} -> {} ({:.1f}%)", k.getSourceID(), k.getEventID(), f, 100.f * ((float)f / (float)vertexInfo.size()));
      // }
      LOGP(info, "received {} unique vertices", seenMCEvent.size());
      for (const auto& vtx : vertexInfo) {
        if (!vtx.mainLabel.isValid() || vtx.bcInROF < 0 || vtx.event < 0) {
          continue;
        }
        mcTree->GetEvent(vtx.event);
        hVtxMCx->Fill(mcEvent->GetX(), vtx.vertex.getX());
        hVtxMCy->Fill(mcEvent->GetY(), vtx.vertex.getY());
        hVtxMCz->Fill(mcEvent->GetZ(), vtx.vertex.getZ());
        if (seenMCEvent[vtx.mainLabel] > 1) {
          pBCPurityDup->Fill(vtx.bcInROF, vtx.purity);
        } else {
          hReVtxZ->Fill(vtx.vertex.getZ());
          hBCVtxNum->Fill(vtx.bcInROF);
          hBCVtxZNum->Fill(vtx.bcInROF, vtx.vertex.getZ());
          pBCPurity->Fill(vtx.bcInROF, vtx.purity);
        }
      }

      auto hBCVtxNumClone = (TH1F*)hBCVtxNum->Clone();
      hBCVtxNumClone->SetTitle("unique Vertex;bcInROF;efficiency");
      hBCVtxNum->Divide(hBCVtxNum, hBCVtxDen, 1., 1., "b");

      auto hBCVtxZNumClone = (TH2F*)hBCVtxZNum->Clone();
      hBCVtxZNumClone->SetTitle("unique Vertex;bcInROF;vtx.z;efficiency");
      hBCVtxZNumClone->Divide(hBCVtxZNum, hBCVtxZDen, 1., 1., "b");

      hReVtxZ->Sumw2();
      hReVtxZ->SetLineColor(kRed);

      auto c = new TCanvas;
      c->Divide(3, 2);

      c->cd(1);
      auto hRatioVtxZ = new TRatioPlot(hReVtxZ, hMCVtxZ);
      hRatioVtxZ->Draw();
      hRatioVtxZ->GetUpperPad()->cd();
      TLegend* legend = new TLegend(0.3, 0.7, 0.7, 0.85);
      legend->SetHeader(Form("MC=%.0f Reco=%.0f", hMCVtxZ->GetEntries(), hReVtxZ->GetEntries()));
      legend->AddEntry(hReVtxZ, "Reco", "l");
      legend->AddEntry(hMCVtxZ, "MC", "le");
      legend->Draw();
      gPad->Update();
      double max1 = hReVtxZ->GetMaximum();
      double max2 = hMCVtxZ->GetMaximum();
      double maxY = std::max(max1, max2);
      hReVtxZ->GetYaxis()->SetRangeUser(0, maxY * 1.1);

      c->cd(2);
      gPad->DrawFrame(-0.5, 1e-3, roFbins - 0.5, 1.1, "Vertex ;bcInROF;eff.");
      hBCVtxNum->Draw("histe;same");

      c->cd(3);
      gPad->DrawFrame(-0.5, 1e-3, roFbins - 0.5, 1.1, "Purity;bcInROF;<purity>");
      pBCPurity->Draw();
      pBCPurityDup->Draw("same");
      c->Draw();

      c->cd(4);
      hBCVtxDen->Draw();
      c->cd(5);
      hBCVtxNumClone->Draw();
      c->cd(6);
      hBCVtxZNumClone->Draw();
      c->Draw();

      c = new TCanvas;
      c->Divide(3, 1);
      c->cd(1);
      hVtxMCx->Draw("colz");
      c->cd(2);
      hVtxMCy->Draw("colz");
      c->cd(3);
      hVtxMCz->Draw("colz");
      c->Draw();
    }
    //////////////////////
    // Fake clusters
    if (write) {
      const int nby{4}, nbz{7};
      double ybins[nby + 1], zbins[nbz + 1];
      for (int i{0}; i < nby + 1; ++i) {
        ybins[i] = (4 + i) - 0.5;
      }
      for (int i{0}; i < nbz + 1; ++i) {
        zbins[i] = (0 + i) - 0.5;
      }
      auto hFakVal = new TH3D("fakClsVal", "Fake cluster attachment;#it{p}_{T} (GeV/#it{c});NCls;Fake;(fake-cluster rate)", nb, xbins, nby, ybins, nbz, zbins);
      auto hFakMig = new TH3D("fakClsMig", "Fake cluster attachment;#it{p}_{T} (GeV/#it{c});NCls;Fake;(fake-cluster rate)", nb, xbins, nby, ybins, nbz, zbins);

      for (auto& event : info) {
        for (auto& part : event) {
          if (!part.isReconstructable()) {
            continue;
          }

          const auto& trk = part.track;
          for (int iL{0}; iL < 7; ++iL) {
            if (!trk.hasHitOnLayer(iL) || !trk.isFakeOnLayer(iL) || (part.clusters & (0x1 << iL)) == 0) {
              continue;
            }
            // TODO: figure out how to find hit migration
            // if (trk.hasHitInNextROF()) {
            //   hFakMig->Fill(trk.getPt(), trk.getNClusters(), iL);
            // } else {
            //   hFakVal->Fill(trk.getPt(), trk.getNClusters(), iL);
            // }
          }
        }
      }

      hFakMig->Write();
      hFakVal->Write();
    }
    if (fOut) {
      fOut->Close();
    }
  } else {
    auto fWO = TFile::Open("checkDROF_wo.root");
    auto f = TFile::Open("checkDROF_w.root");
    plotHistos(fWO, f, "");
    plotHistos(fWO, f, "Val");
    plotHistos(fWO, f, "Mig");
  }
}

void plotHistos(TFile* fWO, TFile* f, const char* append)
{
  TLegend* leg;
  TH1* h;
  const int woStyle = 1001;
  const int wStyle = 3003;
  const int ww{3840}, hh{2160};

  const char* titlename = "";
  if (strcmp(append, "Val") == 0) {
    titlename = ", Validation region";
  } else if (strcmp(append, "Mig") == 0) {
    titlename = ", Migration region";
  }

  auto hWODen2 = fWO->Get<TH2D>(Form("den%s", append));
  hWODen2->SetName(Form("%s_wo", hWODen2->GetName()));
  auto hWONum2 = fWO->Get<TH2D>(Form("num%s", append));
  hWONum2->SetName(Form("%s_wo", hWONum2->GetName()));
  auto hWOFak2 = fWO->Get<TH2D>(Form("fak%s", append));
  hWOFak2->SetName(Form("%s_wo", hWOFak2->GetName()));
  auto hWOSum2 = fWO->Get<TH2D>(Form("sum%s", append));
  hWOSum2->SetName(Form("%s_wo", hWOSum2->GetName()));
  auto hWOMultiFak2 = fWO->Get<TH2D>(Form("multiFak%s", append));
  hWOMultiFak2->SetName(Form("%s_wo", hWOMultiFak2->GetName()));

  auto hWODenMC2 = fWO->Get<TH2D>(Form("denMC%s", append));
  hWODenMC2->SetName(Form("%s_wo", hWODenMC2->GetName()));
  auto hWONumMC2 = fWO->Get<TH2D>(Form("numMC%s", append));
  hWONumMC2->SetName(Form("%s_wo", hWONumMC2->GetName()));
  auto hWOFakMC2 = fWO->Get<TH2D>(Form("fakMC%s", append));
  hWOFakMC2->SetName(Form("%s_wo", hWOFakMC2->GetName()));
  auto hWOSumMC2 = fWO->Get<TH2D>(Form("sumMC%s", append));
  hWOSumMC2->SetName(Form("%s_wo", hWOSumMC2->GetName()));
  auto hWOMultiFakMC2 = fWO->Get<TH2D>(Form("multiFakMC%s", append));
  hWOMultiFakMC2->SetName(Form("%s_wo", hWOMultiFakMC2->GetName()));

  auto hWOBCTracksDen2 = fWO->Get<TH2F>(Form("hBCTracksDen%s", append));
  hWOBCTracksDen2->SetName(Form("%s_wo", hWOBCTracksDen2->GetName()));
  auto hWOBCTracksNum2 = fWO->Get<TH2F>(Form("hBCTracksNum%s", append));
  hWOBCTracksNum2->SetName(Form("%s_wo", hWOBCTracksNum2->GetName()));
  auto hWOBCTracksFake2 = fWO->Get<TH2F>(Form("hBCTracksFake%s", append));
  hWOBCTracksFake2->SetName(Form("%s_wo", hWOBCTracksFake2->GetName()));
  auto hWOBCTracksSum2 = fWO->Get<TH2F>(Form("hBCTracksSum%s", append));
  hWOBCTracksSum2->SetName(Form("%s_wo", hWOBCTracksSum2->GetName()));

  auto setColor = [](TH1* h, EColor c) {
    h->SetLineColor(c);
    h->SetMarkerColor(c);
  };
  auto hDen2 = f->Get<TH2D>(Form("den%s", append));
  setColor(hDen2, kBlack);
  auto hNum2 = f->Get<TH2D>(Form("num%s", append));
  setColor(hNum2, kCyan);
  auto hFak2 = f->Get<TH2D>(Form("fak%s", append));
  setColor(hFak2, kOrange);
  auto hSum2 = f->Get<TH2D>(Form("sum%s", append));
  setColor(hSum2, kGray);
  auto hMultiFak2 = f->Get<TH2D>(Form("multiFak%s", append));
  setColor(hMultiFak2, kMagenta);

  auto hDenMC2 = f->Get<TH2D>(Form("denMC%s", append));
  setColor(hDenMC2, kBlack);
  auto hNumMC2 = f->Get<TH2D>(Form("numMC%s", append));
  setColor(hNumMC2, kCyan);
  auto hFakMC2 = f->Get<TH2D>(Form("fakMC%s", append));
  setColor(hFakMC2, kOrange);
  auto hSumMC2 = f->Get<TH2D>(Form("sumMC%s", append));
  setColor(hSumMC2, kGray);
  auto hMultiFakMC2 = f->Get<TH2D>(Form("multiFakMC%s", append));
  setColor(hMultiFakMC2, kMagenta);

  auto hBCTracksDen2 = f->Get<TH2F>(Form("hBCTracksDen%s", append));
  setColor(hBCTracksDen2, kBlack);
  auto hBCTracksNum2 = f->Get<TH2F>(Form("hBCTracksNum%s", append));
  setColor(hBCTracksNum2, kCyan);
  auto hBCTracksFake2 = f->Get<TH2F>(Form("hBCTracksFake%s", append));
  setColor(hBCTracksFake2, kOrange);
  auto hBCTracksSum2 = f->Get<TH2F>(Form("hBCTracksSum%s", append));
  setColor(hBCTracksSum2, kGray);

  int k = 0;
  TCanvas *cEff = nullptr, *cBC = nullptr, *cCont = nullptr, *cRatio = nullptr;
  {
    auto plotTrkEff = [&](int i, int j) {
      auto hWONum = hWONumMC2->ProjectionX(Form("%s_%d_%d_eff_px", hWONumMC2->GetName(), i, j), i, j);
      auto hWODen = hWODenMC2->ProjectionX(Form("%s_%d_%d_eff_px", hWODenMC2->GetName(), i, j), 0, 5);
      auto hWOFak = hWOFakMC2->ProjectionX(Form("%s_%d_%d_eff_px", hWOFakMC2->GetName(), i, j), i, j);
      auto hWOMultiFak = hWOMultiFakMC2->ProjectionX(Form("%s_%d_%d_eff_px", hWOMultiFakMC2->GetName(), i, j), i, j);
      auto hWOSum = (TH1D*)hWONum->Clone(Form("%s_sum_eff__%d", hWONum->GetName(), j));
      hWOSum->Add(hWOFak);

      hWOSum->Divide(hWOSum, hWODen, 1., 1., "B");
      hWOSum->SetFillColorAlpha(hWOSum2->GetLineColor(), 0.5);
      hWOSum->SetFillStyle(woStyle);
      hWOSum->Draw("histe;same");

      hWONum->Divide(hWONum, hWODen, 1., 1., "B");
      hWONum->SetFillColorAlpha(hWONum2->GetLineColor(), 0.5);
      hWONum->SetFillStyle(woStyle);
      hWONum->Draw("histe;same");

      hWOFak->Divide(hWOFak, hWODen, 1., 1., "B");
      hWOFak->SetFillColorAlpha(hWOFak2->GetLineColor(), 0.5);
      hWOFak->SetFillStyle(woStyle);
      hWOFak->Draw("histe;same");

      hWOMultiFak->Divide(hWOMultiFak, hWODen, 1., 1., "B");
      hWOMultiFak->SetLineColor(hWOMultiFak2->GetLineColor());
      hWOMultiFak->SetFillColorAlpha(hWOMultiFak2->GetLineColor(), 0.5);
      hWOMultiFak->SetFillStyle(woStyle);
      // hWOMultiFak->Draw("histe;same");

      auto hNum = hNum2->ProjectionX(Form("%s_%d_%d_eff_px", hNumMC2->GetName(), i, j), i, j);
      auto hDen = hDen2->ProjectionX(Form("%s_%d_%d_eff_px", hDenMC2->GetName(), i, j), 0, 5);
      auto hFak = hFak2->ProjectionX(Form("%s_%d_%d_eff_px", hFakMC2->GetName(), i, j), i, j);
      auto hMultiFak = hMultiFak2->ProjectionX(Form("%s_%d_%d_px", hMultiFakMC2->GetName(), i, j), i, j);
      auto hSum = (TH1D*)hNum->Clone(Form("%s_sum_eff_%d", hNum->GetName(), j));
      hSum->Add(hFak);

      hSum->Divide(hSum, hDen, 1., 1., "B");
      hSum->SetFillColor(hSum2->GetLineColor());
      hSum->SetLineColor(hSum2->GetLineColor());
      hSum->SetFillStyle(wStyle);
      hSum->Draw("histe;same");

      hNum->Divide(hNum, hDen, 1., 1., "B");
      hNum->SetFillColor(hNum2->GetLineColor());
      hNum->SetLineColor(hNum2->GetLineColor());
      hNum->SetFillStyle(wStyle);
      hNum->Draw("histe;same");

      hFak->Divide(hFak, hDen, 1., 1., "B");
      hFak->SetFillColor(hFak2->GetLineColor());
      hFak->SetLineColor(hFak2->GetLineColor());
      hFak->SetFillStyle(wStyle);
      hFak->Draw("histe;same");

      hMultiFak->Divide(hMultiFak, hDen, 1., 1., "B");
      hMultiFak->SetLineColor(hMultiFak2->GetLineColor());
      hMultiFak->SetFillColor(hMultiFak2->GetLineColor());
      hMultiFak->SetFillStyle(wStyle);
      // hMultiFak->Draw("histe;same");

      if (i == 1 && i == j) {
        leg = new TLegend(0.1, 0.1, 0.9, 0.9);
        leg->AddEntry((TObject*)0, "deltaRof=0", "");
        leg->AddEntry(hWOSum, "sum");
        leg->AddEntry(hWONum, "good");
        leg->AddEntry(hWOFak, "fake");
        // leg->AddEntry(hWOMultiFak, "multifake");
        leg->AddEntry((TObject*)0, "deltaRof=1", "");
        leg->AddEntry(hSum, "sum");
        leg->AddEntry(hNum, "good");
        leg->AddEntry(hFak, "fake");
        // leg->AddEntry(hMultiFak, "multifake");
      }
    };

    cEff = new TCanvas(Form("pteff%s", append), "", ww, hh);
    cEff->Divide(3, 2);
    k = 0;
    for (int i{1}; i <= 4; ++i) {
      if (i == 3) {
        ++k;
      }
      cEff->cd(i + k);
      h = gPad->DrawFrame(
        0.02, 0, 10, 1.02,
        Form("Tracking Efficiency #times Fraction (7 MC hits, %d-point "
             "tracks%s);#it{p}_{T,MC} GeV/#it{c};eff. (fake-rate)",
             3 + i, titlename));
      h->GetXaxis()->SetTitleOffset(1.4);

      plotTrkEff(i, i);

      gPad->SetLogx();
      gPad->SetGrid();
      gPad->RedrawAxis("g");
    }
    cEff->cd(3);
    h = gPad->DrawFrame(
      0.02, 0, 10, 1.02,
      Form("Tracking Efficiency (7 MC hits, all point "
           "tracks%s);#it{p}_{T,MC} GeV/#it{c};eff. (fake-rate)",
           titlename));
    h->GetXaxis()->SetTitleOffset(1.4);

    plotTrkEff(1, 4);

    gPad->SetLogx();
    gPad->SetGrid();
    gPad->RedrawAxis("g");

    cEff->cd(6);
    leg->Draw();
  }

  {
    auto plotRatios = [&](int i, int j, TPad* upper, TPad* lower) {
      auto hWONum = hWONumMC2->ProjectionX(Form("%s_%d_%d_ratio_px", hWONumMC2->GetName(), i, j), i, j);
      auto hWOFak = hWOFakMC2->ProjectionX(Form("%s_%d_%d_ratio_px", hWOFakMC2->GetName(), i, j), i, j);

      hWONum->SetFillColorAlpha(hWONum2->GetLineColor(), 0.5);
      hWONum->SetLineColor(hWONum2->GetLineColor());
      // hWONum->SetFillStyle(woStyle);

      hWOFak->SetFillColorAlpha(hWOFak2->GetLineColor(), 0.5);
      hWOFak->SetLineColor(hWOFak2->GetLineColor());
      // hWOFak->SetFillStyle(woStyle);

      auto hNum = hNum2->ProjectionX(Form("%s_%d_%d_ratio_px", hNumMC2->GetName(), i, j), i, j);
      auto hFak = hFak2->ProjectionX(Form("%s_%d_%d_ratio_px", hFakMC2->GetName(), i, j), i, j);

      hNum->SetFillColor(hNum2->GetLineColor());
      hNum->SetLineColor(hNum2->GetLineColor());
      // hNum->SetFillStyle(wStyle);

      hFak->SetFillColor(hFak2->GetLineColor());
      hFak->SetLineColor(hFak2->GetLineColor());
      // hFak->SetFillStyle(wStyle);
      //
      upper->cd();
      upper->SetLogx();
      upper->SetGrid();
      hWONum->Draw("hist");
      hWOFak->Draw("hist same");
      hNum->Draw("hist same");
      hFak->Draw("hist same");
      double ymax = 1.1 * std::max({hNum->GetMaximum(), hFak->GetMaximum(), hWONum->GetMaximum(), hWOFak->GetMaximum()});
      hWONum->GetYaxis()->SetRangeUser(0, ymax);
      gPad->RedrawAxis("g");

      auto rNum = (TH1*)hNum->Clone(Form("rNum_%s_%d_%d", hNum->GetName(), i, j));
      auto rFak = (TH1*)hFak->Clone(Form("rFak_%s_%d_%d", hFak->GetName(), i, j));
      rNum->GetYaxis()->SetTitle("(deltaRof=1) / (deltaRof=0)");
      rNum->Divide(hWONum);
      rFak->Divide(hWOFak);

      // rNum->SetMarkerStyle(20);
      // rFak->SetMarkerStyle(21);
      rNum->SetLineWidth(2);
      rFak->SetLineWidth(2);
      rNum->SetFillStyle(0);
      rFak->SetFillStyle(0);
      setColor(rNum, kBlue);
      setColor(rFak, kRed);
      double ymin = std::min(rNum->GetMinimum(0.0), rFak->GetMinimum(0.0));
      ymax = std::max(rNum->GetMaximum(), rFak->GetMaximum());
      double ypad = 0.1 * (ymax - ymin);
      ymin -= ypad;
      ymax += ypad;

      lower->cd();
      lower->SetLogx();
      lower->SetGrid();
      rNum->GetYaxis()->SetRangeUser(ymin, ymax);
      rNum->Draw("hist");
      rFak->Draw("hist;same");
      gPad->RedrawAxis("g");

      if (i == 1 && i == j) {
        leg = new TLegend(0.1, 0.1, 0.9, 0.9);
        leg->AddEntry((TObject*)0, "deltaRof=0", "");
        leg->AddEntry(hWONum, "good");
        leg->AddEntry(hWOFak, "fake");
        leg->AddEntry((TObject*)0, "deltaRof=1", "");
        leg->AddEntry(hNum, "good");
        leg->AddEntry(hFak, "fake");
        leg->AddEntry((TObject*)0, "Ratios", "");
        leg->AddEntry(rNum, "good", "l");
        leg->AddEntry(rFak, "fake", "l");
      }
    };

    cRatio = new TCanvas(Form("ptratio%s", append), "", ww, hh);
    cRatio->Divide(3, 2);
    k = 0;
    for (int i{1}; i <= 4; ++i) {
      if (i == 3) {
        ++k;
      }
      cRatio->cd(i + k);
      TPad* up = new TPad(Form("up%d", k), "", 0, 0.5, 1, 1);
      TPad* dn = new TPad(Form("dn%d", k), "", 0, 0, 1, 0.5);
      up->SetBottomMargin(0);
      dn->SetTopMargin(0);
      up->Draw();
      dn->Draw();

      plotRatios(i, i, up, dn);
    }
    cRatio->cd(3);
    TPad* up = new TPad(Form("up_e_%d", k), "", 0, 0.5, 1, 1);
    TPad* dn = new TPad(Form("dn_e_%d", k), "", 0, 0, 1, 0.5);
    up->SetBottomMargin(0);
    dn->SetTopMargin(0);
    up->Draw();
    dn->Draw();
    plotRatios(1, 4, up, dn);

    cRatio->cd(6);
    leg->Draw();
  }

  {
    auto plotTrkCont = [&](int i, int j) {
      auto hWONum = hWONum2->ProjectionX(Form("%s_%d_%d_cont_px", hWONum2->GetName(), i, j), i, j);
      auto hWODen = hWODen2->ProjectionX(Form("%s_%d_%d_cont_px", hWODen2->GetName(), i, j), 0, 5);
      auto hWOFak = hWOFak2->ProjectionX(Form("%s_%d_%d_cont_px", hWOFak2->GetName(), i, j), i, j);
      auto hWOMultiFak = hWOMultiFak2->ProjectionX(Form("%s_%d_%d_cont_px", hWOMultiFak2->GetName(), i, j), i, j);
      auto hWOSum = (TH1D*)hWONum->Clone(Form("%s_sum_cont_%d", hWONum->GetName(), j));
      hWOSum->Add(hWOFak);

      hWOSum->Divide(hWOSum, hWODen, 1., 1., "B");
      hWOSum->SetFillColorAlpha(hWOSum2->GetLineColor(), 0.5);
      hWOSum->SetFillStyle(woStyle);
      // hWOSum->Draw("histe;same");

      hWONum->Divide(hWONum, hWODen, 1., 1., "B");
      hWONum->SetFillColorAlpha(hWONum2->GetLineColor(), 0.5);
      hWONum->SetFillStyle(woStyle);
      hWONum->Draw("histe;same");

      hWOFak->Divide(hWOFak, hWODen, 1., 1., "B");
      hWOFak->SetFillColorAlpha(hWOFak2->GetLineColor(), 0.5);
      hWOFak->SetFillStyle(woStyle);
      hWOFak->Draw("histe;same");

      hWOMultiFak->Divide(hWOMultiFak, hWODen, 1., 1., "B");
      hWOMultiFak->SetLineColor(hWOMultiFak2->GetLineColor());
      hWOMultiFak->SetFillColorAlpha(hWOMultiFak2->GetLineColor(), 0.5);
      hWOMultiFak->SetFillStyle(woStyle);
      // hWOMultiFak->Draw("histe;same");

      auto hNum = hNum2->ProjectionX(Form("%s_%d_%d_cont_px", hNum2->GetName(), i, j), i, j);
      auto hDen = hDen2->ProjectionX(Form("%s_%d_%d_cont_px", hDen2->GetName(), i, j), 0, 5);
      auto hFak = hFak2->ProjectionX(Form("%s_%d_%d_cont_px", hFak2->GetName(), i, j), i, j);
      auto hMultiFak = hMultiFak2->ProjectionX(Form("%s_%d_%d_px", hMultiFak2->GetName(), i, j), i, j);
      auto hSum = (TH1D*)hNum->Clone(Form("%s_sum_cont_%d", hNum->GetName(), j));
      hSum->Add(hFak);

      hSum->Divide(hSum, hDen, 1., 1., "B");
      hSum->SetFillColor(hSum2->GetLineColor());
      hSum->SetFillStyle(wStyle);
      // hSum->Draw("histe;same");

      hNum->Divide(hNum, hDen, 1., 1., "B");
      hNum->SetFillColor(hNum2->GetLineColor());
      hNum->SetFillStyle(wStyle);
      hNum->Draw("histe;same");

      hFak->Divide(hFak, hDen, 1., 1., "B");
      hFak->SetFillColor(hFak2->GetLineColor());
      hFak->SetFillStyle(wStyle);
      hFak->Draw("histe;same");

      hMultiFak->Divide(hMultiFak, hDen, 1., 1., "B");
      hMultiFak->SetLineColor(hMultiFak2->GetLineColor());
      hMultiFak->SetFillColor(hMultiFak2->GetLineColor());
      hMultiFak->SetFillStyle(wStyle);
      // hMultiFak->Draw("histe;same");

      if (i == 1 && i == j) {
        leg = new TLegend(0.1, 0.1, 0.9, 0.9);
        leg->AddEntry((TObject*)0, "deltaRof=0", "");
        leg->AddEntry(hWONum, "good");
        leg->AddEntry(hWOFak, "fake");
        // leg->AddEntry(hWOMultiFak, "multifake");
        leg->AddEntry((TObject*)0, "deltaRof=1", "");
        leg->AddEntry(hNum, "DROF:good");
        leg->AddEntry(hFak, "DROF:fake");
        // leg->AddEntry(hMultiFak, "DROF:multifake");
      }
    };

    cCont = new TCanvas(Form("ptcont%s", append), "", ww, hh);
    cCont->Divide(3, 2);
    k = 0;
    for (int i{1}; i <= 4; ++i) {
      if (i == 3) {
        ++k;
      }
      cCont->cd(i + k);
      h = gPad->DrawFrame(
        0.02, 0, 10, 1.02,
        Form("Tracking Contribution #times Fraction (7 MC hits, %d-point "
             "tracks%s);#it{p}_{T,Reco} GeV/#it{c};contribtution",
             3 + i, titlename));
      h->GetXaxis()->SetTitleOffset(1.4);

      plotTrkCont(i, i);

      gPad->SetLogx();
      gPad->SetGrid();
      gPad->RedrawAxis("g");
    }
    cCont->cd(3);
    h = gPad->DrawFrame(0.02, 0, 10, 1.02,
                        Form("Tracking Contribution (7 MC hits, all point "
                             "tracks%s);#it{p}_{T,Reco} GeV/#it{c};contribution",
                             titlename));
    h->GetXaxis()->SetTitleOffset(1.4);

    plotTrkCont(1, 4);

    gPad->SetLogx();
    gPad->SetGrid();
    gPad->RedrawAxis("g");

    cCont->cd(6);
    leg->Draw();
  }

  {
    auto plotBCEff = [&](int i, int j) {
      auto hWOBCTracksNum = hWOBCTracksNum2->ProjectionX(Form("%s_%d_%d_bc_px", hWOBCTracksNum2->GetName(), i, j), i, j);
      auto hWOBCTracksDen = hWOBCTracksDen2->ProjectionX(Form("%s_%d_%d_bc_px", hWOBCTracksDen2->GetName(), i, j), 0, 5);
      auto hWOBCTracksFake = hWOBCTracksFake2->ProjectionX(Form("%s_%d_%d_bc_px", hWOBCTracksFake2->GetName(), i, j), i, j);
      auto hWOBCTracksSum = (TH1F*)hWOBCTracksNum->Clone(Form("%s_%d_sum", hWOBCTracksNum->GetName(), j));
      hWOBCTracksSum->Add(hWOBCTracksFake);

      hWOBCTracksSum->Divide(hWOBCTracksSum, hWOBCTracksDen, 1., 1., "B");
      hWOBCTracksSum->SetLineColor(hWOSum2->GetLineColor());
      hWOBCTracksSum->SetFillColorAlpha(hWOSum2->GetLineColor(), 0.5);
      hWOBCTracksSum->SetFillStyle(woStyle);
      hWOBCTracksSum->Draw("histe;same");

      hWOBCTracksNum->Divide(hWOBCTracksNum, hWOBCTracksDen, 1., 1., "B");
      hWOBCTracksNum->SetLineColor(hWONum2->GetLineColor());
      hWOBCTracksNum->SetFillColorAlpha(hWONum2->GetLineColor(), 0.5);
      hWOBCTracksNum->SetFillStyle(woStyle);
      hWOBCTracksNum->Draw("histe;same");

      hWOBCTracksFake->Divide(hWOBCTracksFake, hWOBCTracksDen, 1., 1., "B");
      hWOBCTracksFake->SetLineColor(hWOFak2->GetLineColor());
      hWOBCTracksFake->SetFillColorAlpha(hWOFak2->GetLineColor(), 0.5);
      hWOBCTracksFake->SetFillStyle(woStyle);
      hWOBCTracksFake->Draw("histe;same");

      auto hBCTracksNum = hBCTracksNum2->ProjectionX(Form("%s_%d_%d_bc_px", hBCTracksNum2->GetName(), i, j), i, j);
      auto hBCTracksDen = hBCTracksDen2->ProjectionX(Form("%s_%d_%d_bc_px", hBCTracksDen2->GetName(), i, j), 0, 5);
      auto hBCTracksFake = hBCTracksFake2->ProjectionX(Form("%s_%d_%d_bc_px", hBCTracksFake2->GetName(), i, j), i, j);
      auto hBCTracksSum = (TH1F*)hBCTracksNum->Clone(Form("%s_%d_sum", hBCTracksNum->GetName(), j));
      hBCTracksSum->Add(hBCTracksFake);

      hBCTracksSum->Divide(hBCTracksSum, hBCTracksDen, 1., 1., "B");
      hBCTracksSum->SetLineColor(hSum2->GetLineColor());
      hBCTracksSum->SetFillColor(hSum2->GetLineColor());
      hBCTracksSum->SetFillStyle(wStyle);
      hBCTracksSum->Draw("histe;same");

      hBCTracksNum->Divide(hBCTracksNum, hBCTracksDen, 1., 1., "B");
      hBCTracksNum->SetLineColor(hNum2->GetLineColor());
      hBCTracksNum->SetFillColor(hNum2->GetLineColor());
      hBCTracksNum->SetFillStyle(wStyle);
      hBCTracksNum->Draw("histe;same");

      hBCTracksFake->Divide(hBCTracksFake, hBCTracksDen, 1., 1., "B");
      hBCTracksFake->SetLineColor(hFak2->GetLineColor());
      hBCTracksFake->SetFillColor(hFak2->GetLineColor());
      hBCTracksFake->SetFillStyle(wStyle);
      hBCTracksFake->Draw("histe;same");

      if (i == 1 && i == j) {
        leg = new TLegend(0.1, 0.1, 0.9, 0.9);
        leg->AddEntry((TObject*)0, "deltaRof=0", "");
        leg->AddEntry(hWOBCTracksNum, "good");
        leg->AddEntry(hWOBCTracksFake, "fake");
        leg->AddEntry(hWOBCTracksSum, "sum");
        leg->AddEntry((TObject*)0, "deltaRof=1", "");
        leg->AddEntry(hBCTracksNum, "good");
        leg->AddEntry(hBCTracksFake, "fake");
        leg->AddEntry(hBCTracksSum, "sum");
      }
    };

    cBC = new TCanvas(Form("bceff%s", append), "", ww, hh);
    cBC->Divide(3, 2);
    k = 0;
    for (int i{1}; i <= 4; ++i) {
      if (i == 3) {
        ++k;
      }
      cBC->cd(i + k);
      gPad->DrawFrame(-0.5, 0, 200 - 0.5, 1.02,
                      Form("Tracking Efficiency #times Fraction (#it{p}_{T} "
                           "integrated, %d-point "
                           "tracks%s);BC in "
                           "ROF;eff. (fake-rate)",
                           3 + i, titlename));
      plotBCEff(i, i);
      gPad->SetGrid();
      gPad->RedrawAxis("g");
    }
    cBC->cd(3);
    gPad->DrawFrame(-0.5, 0, 200 - 0.5, 1.02,
                    Form("Tracking Efficiency (#it{p}_{T} "
                         "integrated, all point "
                         "tracks%s);BC in "
                         "ROF;eff. (fake-rate)",
                         titlename));
    plotBCEff(1, 4);
    gPad->SetGrid();
    gPad->RedrawAxis("g");

    cBC->cd(6);
    leg->Draw();
  }

  TString outname = TString::Format("trkeff%s.pdf", append);
  cEff->cd();
  cEff->Update();
  cEff->Print(TString::Format("%s(", outname.Data()), "Title:Tracking Efficiency");
  cRatio->cd();
  cRatio->Update();
  cRatio->Print(outname.Data(), "Title:Ratios");
  cCont->cd();
  cCont->Update();
  cCont->Print(outname.Data(), "Title:Contribution");
  cBC->cd();
  cBC->Update();
  cBC->Print(TString::Format("%s)", outname.Data()), "Title:BC");
}
