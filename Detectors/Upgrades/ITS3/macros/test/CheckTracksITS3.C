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

#include <iostream>
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TROOT.h>

#include "DataFormatsITS/TrackITS.h"
#include "TGeoGlobalMagField.h"
#include "Field/MagneticField.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITS3/CompCluster.h"
#endif
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPad.h>
#include "DataFormatsITSMFT/CompCluster.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCEventHeader.h"

using namespace std;

struct ParticleInfo {
  int event;
  int pdg;
  float pt;
  float recpt;
  float eta;
  float phi;
  float pvx;
  float pvy;
  float pvz;
  float dcaxy;
  float dcaz;
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

void CheckTracksITS3(std::string tracfile = "o2trac_its3.root",
                     std::string clusfile = "o2clus_it3.root",
                     std::string kinefile = "o2sim_Kine.root",
                     std::string magfile = "o2sim_grp.root",
                     std::string inputGeom = "o2sim_geometry.root",
                     bool batch = true)
{

  bool isITS3 = true;
  std::string detName = "IT3";
  if (tracfile.find("o2trac_its.root") != std::string::npos && clusfile.find("o2clus_its.root") != std::string::npos) { // we are analysing ITS tracks
    isITS3 = false;
    detName = "ITS";
  }

  gROOT->SetBatch(batch);

  using namespace o2::itsmft;
  using namespace o2::its;

  // Magnetic field
  o2::base::Propagator::initFieldFromGRP(magfile.data());
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  double orig[3] = {0., 0., 0.};
  float bz = field->getBz(orig);
  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
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

  // Clusters
  TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::its3::CompClusterExt>* clusArrITS3 = nullptr;
  std::vector<CompClusterExt>* clusArrITS = nullptr;
  if (isITS3) {
    clusTree->SetBranchAddress(Form("%sClusterComp", detName.data()), &clusArrITS3);
  } else {
    clusTree->SetBranchAddress(Form("%sClusterComp", detName.data()), &clusArrITS);
  }

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress(Form("%sClusterMCTruth", detName.data()), &clusLabArr);

  // Reconstructed tracks
  TFile* file1 = TFile::Open(tracfile.data());
  TTree* recTree = (TTree*)gFile->Get("o2sim");
  std::vector<TrackITS>* recArr = nullptr;
  recTree->SetBranchAddress(Form("%sTrack", detName.data()), &recArr);
  // Track MC labels
  std::vector<o2::MCCompLabel>* trkLabArr = nullptr;
  recTree->SetBranchAddress(Form("%sTrackMCTruth", detName.data()), &trkLabArr);

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

  std::cout << "** Creating particle/clusters correspondance ... " << std::flush;

  for (int frame = 0; frame < clusTree->GetEntriesFast(); frame++) { // Cluster frames
    if (!clusTree->GetEvent(frame))
      continue;

    auto clssize = (clusArrITS3) ? clusArrITS3->size() : clusArrITS->size();
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

      if (isITS3) {
        const o2::its3::CompClusterExt& c = (*clusArrITS3)[iClus];
        auto layer = gman->getLayer(c.getSensorID());
        info[evID][trackID].clusters |= 1 << layer;
      } else {
        const CompClusterExt& c = (*clusArrITS)[iClus];
        auto layer = gman->getLayer(c.getSensorID());
        info[evID][trackID].clusters |= 1 << layer;
      }
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
        float ip[2]{0., 0.};
        info[evID][trackID].track.getImpactParams(info[evID][trackID].pvx, info[evID][trackID].pvy, info[evID][trackID].pvz, bz, ip);
        info[evID][trackID].dcaxy = ip[0];
        info[evID][trackID].dcaz = ip[1];
        info[evID][trackID].recpt = info[evID][trackID].track.getPt();
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

  for (int i = 0; i <= nb; ++i)
    xbins[i] = ptcutl * std::exp(i * a);

  TH1D* h_pt_num = new TH1D("h_pt_num", ";#it{p}_{T} (GeV/#it{c});Number of tracks", nb, xbins);
  h_pt_num->Sumw2();
  TH1D* h_eta_num = new TH1D("h_eta_num", ";#it{#eta};Number of tracks", 60, -3, 3);
  h_eta_num->Sumw2();
  TH1D* h_phi_num = new TH1D("h_phi_num", ";#varphi;Number of tracks", 360, 0., 2 * TMath::Pi());
  h_phi_num->Sumw2();

  TH1D* h_pt_fake = new TH1D("h_pt_fake", ";#it{p}_{T} (GeV/#it{c});Number of fake tracks", nb, xbins);
  h_pt_fake->Sumw2();
  TH1D* h_pt_multifake = new TH1D("h_pt_multifake", ";#it{p}_{T} (GeV/#it{c});Number of multifake tracks", nb, xbins);
  h_pt_multifake->Sumw2();

  TH1D* h_pt_clones = new TH1D("h_pt_clones", ";#it{p}_{T} (GeV/#it{c});Number of cloned tracks", nb, xbins);
  h_pt_clones->Sumw2();

  TH1D* h_pt_den = new TH1D("h_pt_den", ";#it{p}_{T} (GeV/#it{c});Number of generated primary particles", nb, xbins);
  h_pt_den->Sumw2();
  TH1D* h_eta_den = new TH1D("h_eta_den", ";#it{#eta};Number of generated particles", 60, -3, 3);
  h_eta_num->Sumw2();
  TH1D* h_phi_den = new TH1D("h_phi_den", ";#varphi;Number of generated particles", 360, 0., 2 * TMath::Pi());
  h_phi_num->Sumw2();

  TH2D* h_dcaxy_vs_pt = new TH2D("h_dcaxy_vs_pt", ";#it{p}_{T} (GeV/#it{c});DCA_{xy} (#mum)", nb, xbins, 2000, -500., 500.);
  TH2D* h_dcaxy_vs_eta = new TH2D("h_dcaxy_vs_eta", ";#it{#eta};DCA_{xy} (#mum)", 60, -3, 3, 2000, -500., 500.);
  TH2D* h_dcaxy_vs_phi = new TH2D("h_dcaxy_vs_phi", ";#varphi;DCA_{xy} (#mum)", 360, 0., 2 * TMath::Pi(), 2000, -500., 500.);

  TH2D* h_dcaz_vs_pt = new TH2D("h_dcaz_vs_pt", ";#it{p}_{T} (GeV/#it{c});DCA_{z} (#mum)", nb, xbins, 2000, -500., 500.);
  TH2D* h_dcaz_vs_eta = new TH2D("h_dcaz_vs_eta", ";#it{#eta};DCA_{z} (#mum)", 60, -3, 3, 2000, -500., 500.);
  TH2D* h_dcaz_vs_phi = new TH2D("h_dcaz_vs_phi", ";#varphi;DCA_{z} (#mum)", 360, 0., 2 * TMath::Pi(), 2000, -500., 500.);

  TH1D* h_chi2 = new TH1D("h_chi2", ";#chi^{2};Number of tracks", 200, 0., 100.);

  for (auto& evInfo : info) {
    for (auto& part : evInfo) {
      if ((part.clusters & 0x7f) != 0x7f) {
        // part.clusters != 0x3f && part.clusters != 0x3f << 1 &&
        // part.clusters != 0x1f && part.clusters != 0x1f << 1 && part.clusters != 0x1f << 2 &&
        // part.clusters != 0x0f && part.clusters != 0x0f << 1 && part.clusters != 0x0f << 2 && part.clusters != 0x0f << 3) {
        continue;
      }
      if (!part.isPrimary) {
        continue;
      }

      h_pt_den->Fill(part.pt);
      h_eta_den->Fill(part.eta);
      h_phi_den->Fill(part.phi);

      if (part.isReco) {

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

        h_chi2->Fill(part.track.getChi2());

        if (part.isReco > 1) {
          for (int _i{0}; _i < part.isReco - 1; ++_i) {
            h_pt_clones->Fill(part.pt);
          }
        }
      }
      if (part.isFake) {
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

  TH1D* h_pt_eff = (TH1D*)h_pt_num->Clone("h_pt_eff");
  h_pt_eff->Divide(h_pt_num, h_pt_den, 1., 1., "B");
  h_pt_eff->GetYaxis()->SetTitle("Efficiency");
  h_pt_eff->Write();

  file.Close();
  std::cout << " done." << std::endl;
}
