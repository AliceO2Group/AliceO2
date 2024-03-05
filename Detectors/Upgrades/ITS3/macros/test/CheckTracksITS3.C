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

/// \file CheckTracksITS3.C
/// \brief Simple macro to check ITS3 tracks

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TROOT.h>
#include <TCanvas.h>
#include "TEfficiency.h"
#include <TClonesArray.h>
#include <TFile.h>
#include <TH2F.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPad.h>
#include <TTree.h>
#include "TGeoGlobalMagField.h"

#include "DataFormatsITS/TrackITS.h"
#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/TrackReference.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#endif

using namespace std;
using namespace o2::itsmft;
using namespace o2::its;

struct ParticleInfo {
  int event{};
  int pdg{};
  float pt{};
  float recpt{};
  float eta{};
  float phi{};
  float pvx{};
  float pvy{};
  float pvz{};
  float dcaxy{};
  float dcaz{};
  int mother{};
  int first{};
  unsigned short clusters = 0u;
  unsigned char isReco = 0u;
  unsigned char isFake = 0u;
  bool isPrimary = false;
  unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
  o2::its::TrackITS track;
};

#pragma link C++ class ParticleInfo + ;

void CheckTracksITS3(const std::string& tracfile = "o2trac_its3.root",
                     const std::string& clusfile = "o2clus_it3.root",
                     const std::string& kinefile = "o2sim_Kine.root",
                     const std::string& magfile = "o2sim_grp.root",
                     const std::string& inputGeom = "o2sim_geometry.root",
                     bool batch = true)
{
  gROOT->SetBatch(batch);

  // Magnetic field and Propagator
  o2::base::Propagator::initFieldFromGRP(magfile);
  float bz = o2::base::Propagator::Instance()->getNominalBz();

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();

  // MC tracks
  TFile::Open(kinefile.data());
  TTree* mcTree = (TTree*)gFile->Get("o2sim");
  mcTree->SetBranchStatus("*", 0); // disable all branches
  mcTree->SetBranchStatus("MCTrack*", 1);
  mcTree->SetBranchStatus("MCEventHeader*", 1);

  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree->SetBranchAddress("MCTrack", &mcArr);
  o2::dataformats::MCEventHeader* mcEvent = nullptr;
  mcTree->SetBranchAddress("MCEventHeader.", &mcEvent);

  // Clusters
  TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::itsmft::CompClusterExt>* clusArr = nullptr;
  std::vector<CompClusterExt>* clusArrITS = nullptr;
  clusTree->SetBranchAddress("IT3ClusterComp", &clusArr);
  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("IT3ClusterMCTruth", &clusLabArr);

  // Reconstructed tracks
  TFile::Open(tracfile.data());
  TTree* recTree = (TTree*)gFile->Get("o2sim");
  std::vector<TrackITS>* recArr = nullptr;
  recTree->SetBranchAddress("IT3Track", &recArr);
  // Track MC labels
  std::vector<o2::MCCompLabel>* trkLabArr = nullptr;
  recTree->SetBranchAddress("IT3TrackMCTruth", &trkLabArr);

  std::cout << "** Filling particle table ... " << std::flush;
  int lastEventIDcl = -1, cf = 0;
  int nev = mcTree->GetEntriesFast();
  std::vector<std::vector<ParticleInfo>> info(nev);
  for (int n = 0; n < nev; n++) { // loop over MC events
    mcTree->GetEvent(n);
    info[n].resize(mcArr->size());
    for (unsigned int mcI{0}; mcI < mcArr->size(); ++mcI) {
      auto part = mcArr->at(mcI);
      info[n][mcI].pvx = mcEvent->GetX();
      info[n][mcI].pvy = mcEvent->GetY();
      info[n][mcI].pvz = mcEvent->GetZ();
      info[n][mcI].event = n;
      info[n][mcI].pdg = part.GetPdgCode();
      info[n][mcI].pt = part.GetPt();
      info[n][mcI].phi = part.GetPhi();
      info[n][mcI].eta = part.GetEta();
      info[n][mcI].isPrimary = part.isPrimary();
    }
  }
  std::cout << "done." << std::endl;

  std::cout << "** Creating particle/clusters correspondance ... "
            << std::flush;

  for (int frame = 0; frame < clusTree->GetEntriesFast();
       frame++) { // Cluster frames
    if (clusTree->GetEvent(frame) == 0) {
      continue;
    }

    auto clssize = clusArr->size();
    std::cout << clssize << std::endl;

    for (unsigned int iClus{0}; iClus < clssize; ++iClus) {
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
  ULong_t unaccounted{0}, good{0}, fakes{0}, total{0};
  for (int frame = 0; frame < recTree->GetEntriesFast();
       frame++) { // Cluster frames
    if (recTree->GetEvent(frame) == 0) {
      continue;
    }
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
        float ip[2]{0., 0.};
        info[evID][trackID].track.getImpactParams(info[evID][trackID].pvx, info[evID][trackID].pvy, info[evID][trackID].pvz, bz, ip);
        info[evID][trackID].dcaxy = ip[0];
        info[evID][trackID].dcaz = ip[1];
        info[evID][trackID].recpt = info[evID][trackID].track.getPt();
      }

      fakes += static_cast<ULong_t>(fake);
      good += static_cast<ULong_t>(!fake);
    }
  }
  std::cout << "done." << std::endl;

  std::cout << "** Some statistics:" << std::endl;
  std::cout << "\t- Total number of tracks: " << total << std::endl;
  std::cout << "\t- Total number of tracks not corresponding to particles: "
            << unaccounted << " (" << unaccounted * 100. / total << "%)"
            << std::endl;
  std::cout << "\t- Total number of fakes: " << fakes << " ("
            << fakes * 100. / total << "%)" << std::endl;
  std::cout << "\t- Total number of good: " << good << " ("
            << good * 100. / total << "%)" << std::endl;

  int nb = 100;
  double xbins[nb + 1], ptcutl = 0.01, ptcuth = 10.;
  double a = std::log(ptcuth / ptcutl) / nb;

  for (int i = 0; i <= nb; ++i) {
    xbins[i] = ptcutl * std::exp(i * a);
  }

  auto* h_pt_num = new TH1D("h_pt_num", ";#it{p}_{T} (GeV/#it{c});Number of tracks", nb, xbins);
  auto* h_pt_den = new TH1D("h_pt_den", ";#it{p}_{T} (GeV/#it{c});Number of generated primary particles", nb, xbins);
  auto* h_pt_eff = new TEfficiency("h_pt_eff", "Tracking Efficiency;#it{p}_{T} (GeV/#it{c});Eff.", nb, xbins);

  auto* h_eta_num = new TH1D("h_eta_num", ";#it{#eta};Number of tracks", 60, -3, 3);
  auto* h_eta_den = new TH1D("h_eta_den", ";#it{#eta};Number of generated particles", 60, -3, 3);
  auto* h_eta_eff = new TEfficiency("h_eta_eff", "Tracking Efficiency;#it{#eta};Eff.", 60, -3, 3);

  auto* h_phi_num = new TH1D("h_phi_num", ";#varphi;Number of tracks", 360, 0., 2 * TMath::Pi());
  auto* h_phi_den = new TH1D("h_phi_den", ";#varphi;Number of generated particles", 360, 0., 2 * TMath::Pi());
  auto* h_phi_eff = new TEfficiency("h_phi_eff", "Tracking Efficiency;#varphi;Eff.", 360, 0., 2 * TMath::Pi());

  auto* h_pt_fake = new TH1D("h_pt_fake", ";#it{p}_{T} (GeV/#it{c});Number of fake tracks", nb, xbins);
  auto* h_pt_multifake = new TH1D("h_pt_multifake", ";#it{p}_{T} (GeV/#it{c});Number of multifake tracks", nb, xbins);
  auto* h_pt_clones = new TH1D("h_pt_clones", ";#it{p}_{T} (GeV/#it{c});Number of cloned tracks", nb, xbins);
  auto* h_dcaxy_vs_pt = new TH2D("h_dcaxy_vs_pt", ";#it{p}_{T} (GeV/#it{c});DCA_{xy} (#mum)", nb, xbins, 2000, -500., 500.);
  auto* h_dcaxy_vs_eta = new TH2D("h_dcaxy_vs_eta", ";#it{#eta};DCA_{xy} (#mum)", 60, -3, 3, 2000, -500., 500.);
  auto* h_dcaxy_vs_phi = new TH2D("h_dcaxy_vs_phi", ";#varphi;DCA_{xy} (#mum)", 360, 0., 2 * TMath::Pi(), 2000, -500., 500.);
  auto* h_dcaz_vs_pt = new TH2D("h_dcaz_vs_pt", ";#it{p}_{T} (GeV/#it{c});DCA_{z} (#mum)", nb, xbins, 2000, -500., 500.);
  auto* h_dcaz_vs_eta = new TH2D("h_dcaz_vs_eta", ";#it{#eta};DCA_{z} (#mum)", 60, -3, 3, 2000, -500., 500.);
  auto* h_dcaz_vs_phi = new TH2D("h_dcaz_vs_phi", ";#varphi;DCA_{z} (#mum)", 360, 0., 2 * TMath::Pi(), 2000, -500., 500.);
  auto* h_chi2 = new TH2D("h_chi2", ";#it{p}_{T} (GeV/#it{c});#chi^{2};Number of tracks", nb, xbins, 200, 0., 100.);

  for (auto& evInfo : info) {
    for (auto& part : evInfo) {
      if ((part.clusters & 0x7f) != 0x7f) {
        // part.clusters != 0x3f && part.clusters != 0x3f << 1 &&
        // part.clusters != 0x1f && part.clusters != 0x1f << 1 && part.clusters
        // != 0x1f << 2 && part.clusters != 0x0f && part.clusters != 0x0f << 1
        // && part.clusters != 0x0f << 2 && part.clusters != 0x0f << 3) {
        continue;
      }
      if (!part.isPrimary) {
        continue;
      }

      h_pt_den->Fill(part.pt);
      h_eta_den->Fill(part.eta);
      h_phi_den->Fill(part.phi);

      if (part.isReco != 0u) {

        h_pt_num->Fill(part.pt);
        h_eta_num->Fill(part.eta);
        h_phi_num->Fill(part.phi);
        if (std::abs(part.eta) < 0.5) {
          h_dcaxy_vs_pt->Fill(part.pt, part.dcaxy * 10000);
          h_dcaz_vs_pt->Fill(part.pt, part.dcaz * 10000);
        }
        h_dcaz_vs_eta->Fill(part.eta, part.dcaz * 10000);
        h_dcaxy_vs_eta->Fill(part.eta, part.dcaxy * 10000);
        h_dcaxy_vs_phi->Fill(part.phi, part.dcaxy * 10000);
        h_dcaz_vs_phi->Fill(part.phi, part.dcaz * 10000);

        h_chi2->Fill(part.pt, part.track.getChi2());

        if (part.isReco > 1) {
          for (int _i{0}; _i < part.isReco - 1; ++_i) {
            h_pt_clones->Fill(part.pt);
          }
        }
      }
      if (part.isFake != 0u) {
        h_pt_fake->Fill(part.pt);
        if (part.isFake > 1) {
          for (int _i{0}; _i < part.isFake - 1; ++_i) {
            h_pt_multifake->Fill(part.pt);
          }
        }
      }
    }
  }

  std::cout << "** Streaming output TTree to file ... " << std::flush;
  TFile file("CheckTracksITS3.root", "recreate");
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
  h_pt_num->Write();
  h_eta_num->Write();
  h_phi_num->Write();
  h_pt_den->Write();
  h_eta_den->Write();
  h_phi_den->Write();
  h_pt_multifake->Write();
  h_pt_fake->Write();
  h_dcaxy_vs_pt->Write();
  h_dcaz_vs_pt->Write();
  h_dcaxy_vs_eta->Write();
  h_dcaxy_vs_phi->Write();
  h_dcaz_vs_eta->Write();
  h_dcaz_vs_phi->Write();
  h_pt_clones->Write();
  h_chi2->Write();

  h_pt_eff->SetTotalHistogram(*h_pt_den, "");
  h_pt_eff->SetPassedHistogram(*h_pt_num, "");
  h_pt_eff->SetTitle("Tracking Efficiency;#it{p}_{T} (GeV/#it{c});Eff.");
  h_pt_eff->Write();

  h_phi_eff->SetTotalHistogram(*h_phi_den, "");
  h_phi_eff->SetPassedHistogram(*h_phi_num, "");
  h_phi_eff->SetTitle("Tracking Efficiency;#it{#eta};Eff.");
  h_phi_eff->Write();

  h_eta_eff->SetTotalHistogram(*h_eta_den, "");
  h_eta_eff->SetPassedHistogram(*h_eta_num, "");
  h_eta_eff->SetTitle("Tracking Efficiency;#varphi;Eff.");
  h_eta_eff->Write();

  file.Close();
  std::cout << " done." << std::endl;
}
