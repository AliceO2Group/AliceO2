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
#include <TROOT.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TProfile2D.h>
#include <TLegend.h>
#include <TPad.h>
#include <TTree.h>
#include <Math/GenVector/DisplacementVector3D.h>

#include "Steer/MCKinematicsReader.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITS3Base/SpecsV2.h"
#include "ReconstructionDataFormats/BaseCluster.h"
#include "MathUtils/Utils.h"
#include "ITStracking/MathUtils.h"

#include <optional>
#include <tuple>
#endif

// Refit ITS3 tracks by using the clusters from the OB only then propagate the
// tracks to the IB clusters and calculate their residuals.

constexpr auto mMatCorr{o2::base::Propagator::MatCorrType::USEMatCorrNONE};
constexpr float mMaxStep{2};
constexpr float mMaxSnp{0.95};
o2::its3::TopologyDictionary* mDict;

using Vector3D = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::DefaultCoordinateSystemTag>;
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::DefaultCoordinateSystemTag>;

struct Cluster {
  Vector3D displacement;
  Point3D position;
};

static ULong64_t cProp{0};
static ULong64_t cUpdate{0};
static ULong64_t cTotal{0};
static ULong64_t cGood{0};

std::optional<Point3D> getClusterLoc(const o2::itsmft::CompClusterExt& clus)
{
  auto pattID = clus.getPatternID();
  auto chipID = clus.getSensorID();
  o2::math_utils::Point3D<float> locC;
  if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mDict->isGroup(pattID)) {
    return std::nullopt;
  }
  return mDict->getClusterCoordinates(clus);
}

std::optional<Cluster> propagateTo(o2::track::TrackParCov& trk, const o2::itsmft::CompClusterExt& clus, bool reset, bool update)
{
  ++cTotal;
  auto pattID = clus.getPatternID();
  auto chipID = clus.getSensorID();
  auto locC = getClusterLoc(clus);
  if (!locC.has_value()) {
    return std::nullopt;
  }

  auto trkC = o2::its::GeometryTGeo::Instance()->getMatrixT2L(chipID) ^ (*locC);
  auto gloC = o2::its::GeometryTGeo::Instance()->getMatrixL2G(chipID)(*locC);
  const float alpha = o2::its::GeometryTGeo::Instance()->getSensorRefAlpha(clus.getSensorID());

  const auto bz = o2::base::Propagator::Instance()->getNominalBz();

  if (reset) {
    trk.resetCovariance();
  }

  if (!trk.rotate(alpha) ||
      !o2::base::Propagator::Instance()->propagateToX(trk, trkC.x(), bz, mMaxSnp, mMaxStep, mMatCorr)) {
    ++cProp;
    return std::nullopt;
  }

  if (update) {
    o2::itsmft::Cluster cC{chipID, trkC};
    /* auto chi2 = trk.getPredictedChi2(cC); */
    if (!trk.update(cC)) {
      ++cUpdate;
      return std::nullopt;
    }
  }
  ++cGood;

  const auto trkGlo = trk.getXYZGlo();
  Cluster cluster;
  cluster.displacement = (gloC - trkGlo) * 1e4; // in um
  cluster.position = gloC;
  return cluster;
}

void CheckResidualsITS3(const std::string& dictFileName = "../ccdb/IT3/Calib/ClusterDictionary/snapshot.root",
                        const std::string& collisioncontextFileName = "collisioncontext.root",
                        const std::string& itsTracksFileName = "o2trac_its.root",
                        const std::string& itsClustersFileName = "o2clus_its.root",
                        const std::string& magFileName = "o2sim_grp.root",
                        const std::string& geomFileName = "")
{
  mDict = new o2::its3::TopologyDictionary(dictFileName);

  o2::base::GeometryManager::loadGeometry(geomFileName);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms
  o2::base::Propagator::initFieldFromGRP(magFileName);

  o2::steer::MCKinematicsReader mcReader;
  if (!mcReader.initFromDigitContext(collisioncontextFileName)) {
    LOGP(fatal, "Cannot init MC reader");
  }

  auto fITSTracks = TFile::Open(itsTracksFileName.c_str(), "READ");
  auto tITSTracks = fITSTracks->Get<TTree>("o2sim");
  std::vector<o2::its::TrackITS> tracksITS, *tracksITSPtr{&tracksITS};
  tITSTracks->SetBranchAddress("ITSTrack", &tracksITSPtr);
  std::vector<o2::MCCompLabel> tracksLabITS, *tracksLabITSPtr{&tracksLabITS};
  tITSTracks->SetBranchAddress("ITSTrackMCTruth", &tracksLabITSPtr);
  std::vector<int> tracksITSClusIdx, *tracksITSClusIdxPtr{&tracksITSClusIdx};
  tITSTracks->SetBranchAddress("ITSTrackClusIdx", &tracksITSClusIdxPtr);
  auto fITSClusters = TFile::Open(itsClustersFileName.c_str(), "READ");
  auto tITSClusters = fITSClusters->Get<TTree>("o2sim");
  std::vector<o2::itsmft::CompClusterExt> clusITSArr, *clusITSArrPtr{&clusITSArr};
  tITSClusters->SetBranchAddress("ITSClusterComp", &clusITSArrPtr);
  std::vector<unsigned char> clusPattITSArr, *clusPattITSArrPtr{&clusPattITSArr};
  tITSClusters->SetBranchAddress("ITSClusterPatt", &clusPattITSArrPtr);

  const std::array<float, 4> vOBLength{84.3 / 2.0, 84.3 / 2.0, 147.5 / 2.0, 147.5 / 2.0};
  std::array<TH3F*, 4> hOBDx, hOBDy, hOBDz;
  constexpr int nBins{150};
  for (int i{3}; i < 7; ++i) {
    hOBDx[i - 3] = new TH3F(Form("hOBDx_%d", i), Form("DCA_{x} Layer %d;#varphi (rad);z (cm);#Deltax (#mum)", i), nBins, -TMath::Pi(), TMath::Pi(), nBins, -vOBLength[i - 3], vOBLength[i - 3], nBins, -50, 50);
    ;
    hOBDy[i - 3] = new TH3F(Form("hOBDy_%d", i), Form("DCA_{y} Layer %d;#varphi (rad);z (cm);#Deltay (#mum)", i), nBins, -TMath::Pi(), TMath::Pi(), nBins, -vOBLength[i - 3], vOBLength[i - 3], nBins, -50, 50);
    hOBDz[i - 3] = new TH3F(Form("hOBDz_%d", i), Form("DCA_{z} Layer %d;#varphi (rad);z (cm);#Deltaz (#mum)", i), nBins, -TMath::Pi(), TMath::Pi(), nBins, -vOBLength[i - 3], vOBLength[i - 3], nBins, -50, 50);
  }
  std::array<TH3F*, 6> hIBDx, hIBDy, hIBDz;
  for (int i{0}; i < 6; ++i) {
    hIBDx[i] = new TH3F(Form("hIBDx_%d", i), Form("DCA_{x} Sensor %d;normalized #varphi;normalized z;#Deltax (#mum)", i), nBins, -1.0, 1.0, nBins, -1.0, 1.0, nBins, -500, 500);
    hIBDy[i] = new TH3F(Form("hIBDy_%d", i), Form("DCA_{y} Sensor %d;normalized #varphi;normalized z;#Deltay (#mum)", i), nBins, -1.0, 1.0, nBins, -1.0, 1.0, nBins, -500, 500);
    hIBDz[i] = new TH3F(Form("hIBDz_%d", i), Form("DCA_{z} Sensor %d;normalized #varphi;normalized z;#Deltaz (#mum)", i), nBins, -1.0, 1.0, nBins, -1.0, 1.0, nBins, -2000, 2000);
  }

  ULong64_t cTot{0}, cSkipped{0};
  const Long64_t nEntries = tITSTracks->GetEntries();
  for (Long64_t iEntry{0}; iEntry < nEntries; ++iEntry) {
    tITSTracks->GetEntry(iEntry);
    tITSClusters->GetEntry(iEntry);

    for (unsigned int iTrack{0}; iTrack < tracksITS.size(); ++iTrack) {
      const auto& trk = tracksITS[iTrack];

      if (trk.getNClusters() < 6) {
        continue;
      }

      // are all clusters valid
      const auto& lab = tracksLabITS[iTrack];
      if (!lab.isValid()) {
        continue;
      }

      auto trkOut = trk.getParamOut();
      const int clEntry = trk.getFirstClusterEntry();
      const int ncl = trk.getNumberOfClusters();
      for (int icl = 0; icl != ncl; icl += 1, ++cTot) { // Start from outermost cluster
        const int idx = tracksITSClusIdx[clEntry + icl];
        const auto& clus = clusITSArr[idx];
        const int layer = gman->getLayer(clus.getSensorID());
        const bool needsUpdate = layer > 2; // refit only with IB clusters
        if (const auto res = propagateTo(trkOut, clus, layer == (ncl - 1), needsUpdate)) {
          /* if (const auto res = propagateTo(trkOut, clus, false, false)) { */
          auto cluster = *res;
          if (o2::its3::constants::detID::isDetITS3(clus.getSensorID())) {
            const auto sensorID = o2::its3::constants::detID::getSensorID(clus.getSensorID());
            const bool isTop = sensorID % 2 == 0;
            const float phi = o2::math_utils::to02Pi(cluster.position.Phi());
            const float phi1 = o2::math_utils::to02Pi(((isTop) ? 0.f : 1.f) * TMath::Pi() + std::asin(o2::its3::constants::equatorialGap / 2.f / o2::its3::constants::radii[layer]));
            const float phi2 = o2::math_utils::to02Pi(((isTop) ? 1.f : 2.f) * TMath::Pi() - std::asin(o2::its3::constants::equatorialGap / 2.f / o2::its3::constants::radii[layer]));
            const float nphi = ((phi - phi1) * 2.f) / (phi2 - phi1) - 1.f;
            const float nz = (2.f * cluster.position.Z()) / o2::its3::constants::segment::lengthSensitive;
            hIBDx[sensorID]->Fill(nphi, nz, cluster.displacement.X());
            hIBDy[sensorID]->Fill(nphi, nz, cluster.displacement.Y());
            hIBDz[sensorID]->Fill(nphi, nz, cluster.displacement.Z());
          } else {
            hOBDx[layer - 3]->Fill(cluster.position.Phi(), cluster.position.Z(), cluster.displacement.X());
            hOBDy[layer - 3]->Fill(cluster.position.Phi(), cluster.position.Z(), cluster.displacement.Y());
            hOBDz[layer - 3]->Fill(cluster.position.Phi(), cluster.position.Z(), cluster.displacement.Z());
          }
        } else {
          ++cSkipped;
        }
      }
    }
  }

  LOGP(info, "Skipped {} of {} clusters", cSkipped, cTot);
  LOGP(info, "Total {} Good {} Prop {} Update {}", cTotal, cGood, cProp, cUpdate);

  auto c = new TCanvas();
  c->Divide(4, 2);
  for (int i{0}; i < 4; ++i) {
    c->cd(i + 1);
    hOBDx[i]->Project3D("zx")->Draw("colz");
    c->cd(i + 5);
    hOBDx[i]->Project3D("zy")->Draw("colz");
  }
  c->Draw();
  c->SaveAs("its3_ob_dx.pdf");

  c = new TCanvas();
  c->Divide(4, 2);
  for (int i{0}; i < 4; ++i) {
    c->cd(i + 1);
    hOBDy[i]->Project3D("zx")->Draw("colz");
    c->cd(i + 5);
    hOBDy[i]->Project3D("zy")->Draw("colz");
  }
  c->Draw();
  c->SaveAs("its3_ob_dy.pdf");

  c = new TCanvas();
  c->Divide(4, 2);
  for (int i{0}; i < 4; ++i) {
    c->cd(i + 1);
    hOBDz[i]->Project3D("zx")->Draw("colz");
    c->cd(i + 5);
    hOBDz[i]->Project3D("zy")->Draw("colz");
  }
  c->Draw();
  c->SaveAs("its3_ob_dz.pdf");

  c = new TCanvas();
  c->Divide(6, 2);
  for (int i{0}; i < 6; ++i) {
    c->cd(i + 1);
    hIBDx[i]->Project3D("zx")->Draw("colz");
    c->cd(i + 7);
    hIBDx[i]->Project3D("zy")->Draw("colz");
  }
  c->Draw();
  c->SaveAs("its3_ib_dx.pdf");

  c = new TCanvas();
  c->Divide(6, 2);
  for (int i{0}; i < 6; ++i) {
    c->cd(i + 1);
    hIBDy[i]->Project3D("zx")->Draw("colz");
    c->cd(i + 7);
    hIBDy[i]->Project3D("zy")->Draw("colz");
  }
  c->Draw();
  c->SaveAs("its3_ib_dy.pdf");

  c = new TCanvas();
  c->Divide(6, 2);
  for (int i{0}; i < 6; ++i) {
    c->cd(i + 1);
    hIBDz[i]->Project3D("zx")->Draw("colz");
    c->cd(i + 7);
    hIBDz[i]->Project3D("zy")->Draw("colz");
  }
  c->Draw();
  c->SaveAs("its3_ib_dz.pdf");

  c = new TCanvas();
  c->Divide(6, 3);
  for (int i{0}; i < 6; ++i) {
    c->cd(i + 1);
    hIBDx[i]->Project3DProfile("yx")->Draw("colz");
    c->cd(i + 7);
    hIBDy[i]->Project3DProfile("yx")->Draw("colz");
    c->cd(i + 13);
    hIBDz[i]->Project3DProfile("yx")->Draw("colz");
  }
  c->Draw();
  c->SaveAs("its3_ib_displacement.pdf");
}
