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
#include <TF1.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPad.h>
#include <TLatex.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>

#include "ITSBase/GeometryTGeo.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "DetectorsBase/Propagator.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/O2DatabasePDG.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITS/TrackITS.h"

#endif
#include "DataFormatsITSMFT/CompCluster.h"

using namespace std;

// chi2 PDF with amplitude A, degrees of freedom k, scale s
Double_t chi2_pdf(Double_t* x, Double_t* par)
{
  const Double_t xx = x[0];
  const Double_t A = par[0];
  const Double_t k = par[1];
  const Double_t s = par[2];
  if (xx <= 0.0 || k <= 0.0 || s <= 0.0)
    return 0.0;
  const Double_t coef = 1.0 / (TMath::Power(2.0 * s, k * 0.5) * TMath::Gamma(k * 0.5));
  return A * coef * TMath::Power(xx, k * 0.5 - 1.0) * TMath::Exp(-xx / (2.0 * s));
}

struct ParticleInfo {
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
  bool isPrimary = 0u;
  unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
  o2::its::TrackITS track;
  o2::MCTrack mcTrack;
};

#pragma link C++ class ParticleInfo + ;

void CheckTracksCA(bool doEffStud = true,
                   bool doFakeClStud = false,
                   bool doPullStud = false,
                   bool createOutput = false,
                   std::string tracfile = "o2trac_its.root",
                   std::string magfile = "o2sim_grp.root",
                   std::string clusfile = "o2clus_its.root",
                   std::string kinefile = "o2sim_Kine.root")
{

  using namespace o2::itsmft;
  using namespace o2::its;

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
  std::vector<std::vector<ParticleInfo>> info;
  info.resize(nev);
  TH1D* hZvertex = new TH1D("hZvertex", "Z vertex", 100, -20, 20);
  for (int n = 0; n < nev; n++) { // loop over MC events
    mcTree->GetEvent(n);
    info[n].resize(mcArr->size());
    hZvertex->Fill(mcEvent->GetZ());
    for (unsigned int mcI{0}; mcI < mcArr->size(); ++mcI) {
      const auto part = mcArr->at(mcI);
      if (!o2::O2DatabasePDG::Instance()->GetParticle(part.GetPdgCode())) {
        continue;
      }
      info[n][mcI].event = n;
      info[n][mcI].pdg = part.GetPdgCode();
      info[n][mcI].pvx = mcEvent->GetX();
      info[n][mcI].pvy = mcEvent->GetY();
      info[n][mcI].pvz = mcEvent->GetZ();
      info[n][mcI].pt = part.GetPt();
      info[n][mcI].phi = part.GetPhi();
      info[n][mcI].eta = part.GetEta();
      info[n][mcI].isPrimary = part.isPrimary();
      info[n][mcI].mcTrack = part;
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
        info[evID][trackID].track = recArr->at(iTrack);
        info[evID][trackID].storedStatus = fake;
        static float ip[2]{0., 0.};
        info[evID][trackID].track.getImpactParams(info[evID][trackID].pvx, info[evID][trackID].pvy, info[evID][trackID].pvz, bz, ip);
        info[evID][trackID].dcaxy = ip[0];
        info[evID][trackID].dcaz = ip[1];
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

  TFile* file{nullptr};
  if (createOutput) {
    file = TFile::Open("CheckTracksCA.root", "recreate");
  }

  if (doEffStud) {
    std::cout << "Calculating efficiencies... ";
    const int nb = 100;
    double xbins[nb + 1], ptcutl = 0.01, ptcuth = 10.;
    double a = std::log(ptcuth / ptcutl) / nb;
    for (int i = 0; i <= nb; i++)
      xbins[i] = ptcutl * std::exp(i * a);
    TH1D* num = new TH1D("num", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", nb, xbins);
    num->Sumw2();
    TH1D* numEta = new TH1D("numEta", ";#eta;Number of tracks", 60, -3, 3);
    numEta->Sumw2();
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
        if ((part.clusters & 0x7f) != 0x7f) {
          // part.clusters != 0x3f && part.clusters != 0x3f << 1 &&
          // part.clusters != 0x1f && part.clusters != 0x1f << 1 && part.clusters != 0x1f << 2 &&
          // part.clusters != 0x0f && part.clusters != 0x0f << 1 && part.clusters != 0x0f << 2 && part.clusters != 0x0f << 3) {
          continue;
        }
        if (!part.isPrimary) {
          continue;
        }
        den->Fill(part.pt);
        if (part.isReco) {
          num->Fill(part.pt);
          numEta->Fill(part.eta);
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
    TCanvas* c2 = new TCanvas;
    c2->SetGridx();
    c2->SetGridy();
    hZvertex->DrawClone();

    if (createOutput) {
      sum->Write("total");
      fak->Write("singleFake");
      num->Write("efficiency");
      numEta->Write("etaDist");
      multiFak->Write("multiFake");
      clone->Write("clones");
    }
    std::cout << " done." << std::endl;
  }

  //////////////////////
  // Fake clusters study
  if (doFakeClStud) {
    std::cout << "Creating fake cluster study... ";
    std::vector<TH1I*> histLength, histLength1Fake, histLengthNoCl, histLength1FakeNoCl;
    std::vector<THStack*> stackLength, stackLength1Fake;
    std::vector<TLegend*> legends, legends1Fake;
    histLength.resize(4);
    histLength1Fake.resize(4);
    histLengthNoCl.resize(4);
    histLength1FakeNoCl.resize(4);
    stackLength.resize(4);
    stackLength1Fake.resize(4);
    legends.resize(4);
    legends1Fake.resize(4);

    for (int iH{4}; iH < 8; ++iH) {
      histLength[iH - 4] = new TH1I(Form("trk_len_%d", iH), "#exists cluster", 7, -.5, 6.5);
      histLength[iH - 4]->SetFillColor(kBlue);
      histLength[iH - 4]->SetLineColor(kBlue);
      histLength[iH - 4]->SetFillStyle(3352);
      histLengthNoCl[iH - 4] = new TH1I(Form("trk_len_%d_nocl", iH), "#slash{#exists} cluster", 7, -.5, 6.5);
      histLengthNoCl[iH - 4]->SetFillColor(kRed);
      histLengthNoCl[iH - 4]->SetLineColor(kRed);
      histLengthNoCl[iH - 4]->SetFillStyle(3352);
      stackLength[iH - 4] = new THStack(Form("stack_trk_len_%d", iH), Form("trk_len=%d", iH));
      stackLength[iH - 4]->Add(histLength[iH - 4]);
      stackLength[iH - 4]->Add(histLengthNoCl[iH - 4]);
    }
    for (int iH{4}; iH < 8; ++iH) {
      histLength1Fake[iH - 4] = new TH1I(Form("trk_len_%d_1f", iH), "#exists cluster", 7, -.5, 6.5);
      histLength1Fake[iH - 4]->SetFillColor(kBlue);
      histLength1Fake[iH - 4]->SetLineColor(kBlue);
      histLength1Fake[iH - 4]->SetFillStyle(3352);
      histLength1FakeNoCl[iH - 4] = new TH1I(Form("trk_len_%d_1f_nocl", iH), "#slash{#exists} cluster", 7, -.5, 6.5);
      histLength1FakeNoCl[iH - 4]->SetFillColor(kRed);
      histLength1FakeNoCl[iH - 4]->SetLineColor(kRed);
      histLength1FakeNoCl[iH - 4]->SetFillStyle(3352);
      stackLength1Fake[iH - 4] = new THStack(Form("stack_trk_len_%d_1f", iH), Form("trk_len=%d, 1 Fake", iH));
      stackLength1Fake[iH - 4]->Add(histLength1Fake[iH - 4]);
      stackLength1Fake[iH - 4]->Add(histLength1FakeNoCl[iH - 4]);
    }

    for (const auto& event : info) {
      for (const auto& part : event) {
        int nCl{0};
        for (unsigned int bit{0}; bit < sizeof(part.clusters) * 8; ++bit) {
          nCl += bool(part.clusters & (1 << bit));
        }
        if (nCl < 3) {
          continue;
        }

        auto& track = part.track;
        auto len = track.getNClusters();
        for (int iLayer{0}; iLayer < 7; ++iLayer) {
          if (track.hasHitOnLayer(iLayer)) {
            if (track.isFakeOnLayer(iLayer)) {       // Reco track has fake cluster
              if (part.clusters & (0x1 << iLayer)) { // Correct cluster exists
                histLength[len - 4]->Fill(iLayer);
                if (track.getNFakeClusters() == 1) {
                  histLength1Fake[len - 4]->Fill(iLayer);
                }
              } else {
                histLengthNoCl[len - 4]->Fill(iLayer);
                if (track.getNFakeClusters() == 1) {
                  histLength1FakeNoCl[len - 4]->Fill(iLayer);
                }
              }
            }
          }
        }
      }
    }

    auto canvas = new TCanvas("fc_canvas", "Fake clusters", 1600, 1000);
    canvas->Divide(4, 2);
    for (int iH{0}; iH < 4; ++iH) {
      canvas->cd(iH + 1);
      stackLength[iH]->Draw();
      gPad->BuildLegend();
    }
    for (int iH{0}; iH < 4; ++iH) {
      canvas->cd(iH + 5);
      stackLength1Fake[iH]->Draw();
      gPad->BuildLegend();
    }
    canvas->SaveAs("fakeClusters.png", "recreate");
    std::cout << " done\n";
  }

  if (doPullStud) {
    std::cout << "Creating pull study... ";
    const int nBins{30};
    const float xWidth{10};
    // Pulls
    auto hYPull = new TH1F("hYPull", "Pull Y", nBins, -xWidth, xWidth);
    auto hZPull = new TH1F("hZPull", "Pull Z", nBins, -xWidth, xWidth);
    auto hSPhiPull = new TH1F("hSPhiPull", "Pull Sin(Phi)", nBins, -xWidth, xWidth);
    auto hTglPull = new TH1F("hTglPull", "Pull Tg(Lambda)", nBins, -xWidth, xWidth);
    auto hQoPtPull = new TH1F("hQoPtPull", "Pull Q/Pt", nBins, -xWidth, xWidth);
    // Correlation
    float maxY2{1e-6}, maxZ2{1e-6}, maxSnp2{2e-6}, maxTgl2{2e-6}, max1Pt2{0.01};
    auto hCorYZ = new TH2F("hCorYZ", ";#sigma_{Z}^{2};#sigma_{Y}^{2}", nBins, 0, maxZ2, nBins, 0, maxY2);
    auto hCorYSPhi = new TH2F("hCorYSPhi", ";#sigma_{snp}^{2};#sigma_{Y}^{2}", nBins, 0, maxSnp2, nBins, 0, maxY2);
    auto hCorYTgl = new TH2F("hCorYTgl", ";#sigma_{tgl}^{2};#sigma_{Y}^{2}", nBins, 0, maxTgl2, nBins, 0, maxY2);
    auto hCorYQoPt = new TH2F("hCorYQoPt", ";#sigma_{Q/Pt}^{2};#sigma_{Y}^{2}", nBins, 0, max1Pt2, nBins, 0, maxY2);

    auto hCorZSPhi = new TH2F("hCorZSPhi", ";#sigma_{snp}^{2};#sigma_{Z}^{2}", nBins, 0, maxSnp2, nBins, 0, maxZ2);
    auto hCorZTgl = new TH2F("hCorZTgl", ";#sigma_{tgl}^{2};#sigma_{Z}^{2}", nBins, 0, maxTgl2, nBins, 0, maxZ2);
    auto hCorZQoPt = new TH2F("hCorZQoPt", ";#sigma_{Q/Pt}^{2};#sigma_{Z}^{2}", nBins, 0, max1Pt2, nBins, 0, maxZ2);

    auto hCorSPhiTgl = new TH2F("hCorSPhiTgl", ";#sigma_{tgl}^{2};#sigma_{snp}^{2}", nBins, 0, maxTgl2, nBins, 0, maxSnp2);
    auto hCorSPhiQoPt = new TH2F("hCorSPhiQoPt", ";#sigma_{Q/Pt}^{2};#sigma_{snp}^{2}", nBins, 0, max1Pt2, nBins, 0, maxSnp2);

    auto hCorTglQoPt = new TH2F("hCorTglQoPt", ";#sigma_{Q/Pt}^{2};#sigma_{tgl}^{2}", nBins, 0, max1Pt2, nBins, 0, maxTgl2);

    auto calcMahalanobisDist = [&](const auto& trk, const auto& mc) -> float {
      o2::math_utils::SMatrix<float, o2::track::kNParams, o2::track::kNParams, o2::math_utils::MatRepSym<float, o2::track::kNParams>> cov;
      cov(o2::track::kY, o2::track::kY) = trk.getSigmaY2();
      cov(o2::track::kZ, o2::track::kY) = trk.getSigmaZY();
      cov(o2::track::kZ, o2::track::kZ) = trk.getSigmaZ2();
      cov(o2::track::kSnp, o2::track::kY) = trk.getSigmaSnpY();
      cov(o2::track::kSnp, o2::track::kZ) = trk.getSigmaSnpZ();
      cov(o2::track::kSnp, o2::track::kSnp) = trk.getSigmaSnp2();
      cov(o2::track::kTgl, o2::track::kY) = trk.getSigmaTglY();
      cov(o2::track::kTgl, o2::track::kZ) = trk.getSigmaTglZ();
      cov(o2::track::kTgl, o2::track::kSnp) = trk.getSigmaTglSnp();
      cov(o2::track::kTgl, o2::track::kTgl) = trk.getSigmaTgl2();
      cov(o2::track::kQ2Pt, o2::track::kY) = trk.getSigma1PtY();
      cov(o2::track::kQ2Pt, o2::track::kZ) = trk.getSigma1PtZ();
      cov(o2::track::kQ2Pt, o2::track::kSnp) = trk.getSigma1PtSnp();
      cov(o2::track::kQ2Pt, o2::track::kTgl) = trk.getSigma1PtTgl();
      cov(o2::track::kQ2Pt, o2::track::kQ2Pt) = trk.getSigma1Pt2();
      if (!cov.Invert()) {
        return -1.f;
      }
      o2::math_utils::SVector<float, o2::track::kNParams> trkPar(trk.getParams(), o2::track::kNParams), mcPar(mc.getParams(), o2::track::kNParams);
      auto res = trkPar - mcPar;
      return std::sqrt(ROOT::Math::Similarity(cov, res));
    };
    auto hMahDist = new TH1F("hMahDist", ";Mahalanobis distance;n. entries", 100, 0, 10);
    TF1* fchi = new TF1("fchi", chi2_pdf, 0, 6, 3);
    fchi->SetParNames("A", "k", "s");
    fchi->SetParameter(0, 1);
    fchi->SetParameter(1, 5);
    fchi->SetParameter(2, 1);

    for (const auto& event : info) {
      for (const auto& part : event) {
        if (((part.clusters & 0x7f) != 0x7f) && !part.isPrimary) {
          continue;
        }

        // prepare mc truth parameters
        std::array<float, 3> xyz{(float)part.mcTrack.GetStartVertexCoordinatesX(), (float)part.mcTrack.GetStartVertexCoordinatesY(), (float)part.mcTrack.GetStartVertexCoordinatesZ()};
        std::array<float, 3> pxyz{(float)part.mcTrack.GetStartVertexMomentumX(), (float)part.mcTrack.GetStartVertexMomentumY(), (float)part.mcTrack.GetStartVertexMomentumZ()};
        o2::track::TrackPar mcTrack(xyz, pxyz, TMath::Nint(o2::O2DatabasePDG::Instance()->GetParticle(part.mcTrack.GetPdgCode())->Charge() / 3), false);
        if (!mcTrack.rotate(part.track.getAlpha()) ||
            !o2::base::Propagator::Instance()->propagateTo(mcTrack, part.track.getX())) {
          continue;
        }

        const float sY = part.track.getSigmaY2();
        const float sZ = part.track.getSigmaZ2();
        const float sSnp = part.track.getSigmaSnp2();
        const float sTgl = part.track.getSigmaTgl2();
        const float s1Pt = part.track.getSigma1Pt2();

        hYPull->Fill((part.track.getY() - mcTrack.getY()) / std::sqrt(part.track.getSigmaY2()));
        hZPull->Fill((part.track.getZ() - mcTrack.getZ()) / std::sqrt(part.track.getSigmaZ2()));
        hSPhiPull->Fill((part.track.getSnp() - mcTrack.getSnp()) / std::sqrt(part.track.getSigmaSnp2()));
        hTglPull->Fill((part.track.getTgl() - mcTrack.getTgl()) / std::sqrt(part.track.getSigmaTgl2()));
        hQoPtPull->Fill((part.track.getQ2Pt() - mcTrack.getQ2Pt()) / std::sqrt(part.track.getSigma1Pt2()));

        hCorYZ->Fill(part.track.getSigmaZ2(), part.track.getSigmaY2());
        hCorYSPhi->Fill(part.track.getSigmaSnp2(), part.track.getSigmaY2());
        hCorYTgl->Fill(part.track.getSigmaTgl2(), part.track.getSigmaY2());
        hCorYQoPt->Fill(part.track.getSigma1Pt2(), part.track.getSigmaY2());

        hCorZSPhi->Fill(part.track.getSigmaSnp2(), part.track.getSigmaZ2());
        hCorZTgl->Fill(part.track.getSigmaTgl2(), part.track.getSigmaZ2());
        hCorZQoPt->Fill(part.track.getSigma1Pt2(), part.track.getSigmaZ2());

        hCorSPhiTgl->Fill(part.track.getSigmaTgl2(), part.track.getSigmaSnp2());
        hCorSPhiQoPt->Fill(part.track.getSigma1Pt2(), part.track.getSigmaSnp2());

        hCorTglQoPt->Fill(part.track.getSigma1Pt2(), part.track.getSigmaTgl2());

        hMahDist->Fill(calcMahalanobisDist(part.track, mcTrack));
      }
    }

    // normalise, set axis, fit and draw
    auto doPullCalc = [](TH1F* h) {
      h->Scale(1. / h->Integral("width"));
      h->GetYaxis()->SetRangeUser(1e-5, 1.);
      gPad->SetLogy();
      h->Draw("hist");
      h->Fit("gaus", "QMR", "", -3, 3);
      if (auto f = h->GetFunction("gaus")) {
        f->SetLineColor(kRed);
        f->SetLineWidth(2);
        f->Draw("same");
        const double mean = f->GetParameter(1);
        const double sigma = f->GetParameter(2);
        TLatex lat;
        lat.SetNDC();
        lat.SetTextFont(42);
        lat.SetTextSize(0.04);
        lat.DrawLatex(0.62, 0.85, Form("#mu = %.4f", mean));
        lat.DrawLatex(0.62, 0.79, Form("#sigma = %.4f", sigma));
      }
    };
    hMahDist->Scale(1. / hMahDist->Integral("width"));
    TFitResultPtr fitres = hMahDist->Fit(fchi, "RMQS");

    auto c = new TCanvas("cPull", "", 2000, 1000);
    c->Divide(5, 5);
    c->cd(1);
    doPullCalc(hYPull);
    c->cd(2);
    hCorYZ->Draw("colz");
    c->cd(3);
    hCorYSPhi->Draw("colz");
    c->cd(4);
    hCorYTgl->Draw("colz");
    c->cd(5);
    hCorYQoPt->Draw("colz");

    c->cd(7);
    doPullCalc(hZPull);
    c->cd(8);
    hCorZSPhi->Draw("colz");
    c->cd(9);
    hCorZTgl->Draw("colz");
    c->cd(10);
    hCorZQoPt->Draw("colz");

    c->cd(13);
    doPullCalc(hSPhiPull);
    c->cd(14);
    hCorSPhiTgl->Draw("colz");
    c->cd(15);
    hCorSPhiQoPt->Draw("colz");

    c->cd(19);
    doPullCalc(hTglPull);
    c->cd(20);
    hCorTglQoPt->Draw("colz");

    c->cd();
    const double xlow = 0.0;
    const double xup = 0.4;
    const double ylow = 0.0;
    const double yup = 0.4;
    auto pMahBig = new TPad("pMahBig", "Mahalanobis Distance", xlow, ylow, xup, yup);
    pMahBig->SetFillStyle(4000);
    pMahBig->SetBorderMode(0);
    pMahBig->SetLeftMargin(0.12);
    pMahBig->SetRightMargin(0.02);
    pMahBig->SetBottomMargin(0.12);
    pMahBig->SetTopMargin(0.05);
    pMahBig->Draw();
    pMahBig->cd();
    hMahDist->Draw("hist");
    fchi->SetLineColor(kRed);
    fchi->SetLineWidth(2);
    fchi->Draw("same");
    const Double_t A_fit = fchi->GetParameter(0);
    const Double_t k_fit = fchi->GetParameter(1);
    const Double_t s_fit = fchi->GetParameter(2);
    const Double_t A_err = fchi->GetParError(0);
    const Double_t k_err = fchi->GetParError(1);
    const Double_t s_err = fchi->GetParError(2);
    const Double_t chi2 = fchi->GetChisquare();
    const Int_t ndf = fchi->GetNDF();
    TLatex lat;
    lat.SetNDC();
    lat.SetTextFont(42);
    lat.SetTextSize(0.038);
    lat.SetTextAlign(11);
    const Double_t xText = 0.55;
    Double_t yText = 0.85;
    const Double_t dy = 0.06;
    lat.DrawLatex(xText, yText, Form("A = %.3g #pm %.3g", A_fit, A_err));
    yText -= dy;
    lat.DrawLatex(xText, yText, Form("k (ndf) = %.3f #pm %.3f", k_fit, k_err));
    yText -= dy;
    lat.DrawLatex(xText, yText, Form("scale s = %.3g #pm %.3g", s_fit, s_err));
    yText -= dy;
    lat.DrawLatex(xText, yText, Form("#chi^{2}/ndf = %.2f / %d", chi2, ndf));
    yText -= dy;
    if (fitres.Get()) {
      lat.DrawLatex(xText, yText, Form("Fit status = %d", fitres->Status()));
      yText -= dy;
    }

    c->cd(25);
    doPullCalc(hQoPtPull);
    c->Draw();
    std::cout << " done\n";
  }

  if (createOutput) {
    std::cout << "** Streaming output TTree to file ... " << std::flush;
    TTree tree("ParticleInfo", "ParticleInfo");
    ParticleInfo pInfo;
    tree.Branch("particle", &pInfo);
    for (const auto& event : info) {
      for (const auto& part : event) {
        if (((part.clusters & 0x7f) != 0x7f) && !part.isPrimary) {
          continue;
        }
        pInfo = part;
        tree.Fill();
      }
    }
    tree.Write();
    file->Close();
  }
}
