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

#include "ITSStudies/AvgClusSize.h"


#include "DataFormatsITS/TrackITS.h"
// #include "Framework/CCDBParamSpec.h"
#include "Framework/Task.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/V0.h"
#include "CommonUtils/TreeStreamRedirector.h"
// #include "ITStracking/IOUtils.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSBase/GeometryTGeo.h"

#include <TH1F.h>
#include <TFile.h>
#include <TCanvas.h>

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;
using V0 = o2::dataformats::V0;
using ITSCluster = o2::BaseCluster<float>;
using mask_t = o2::dataformats::GlobalTrackID::mask_t;
using AvgClusSizeStudy = o2::its::study::AvgClusSizeStudy;

// AvgClusSizeStudy::AvgClusSizeStudy(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool isMC) {}// : mDataRequest{dr}, mGGCCDBRequest(gr), mUseMC(isMC)
// {
// }

void AvgClusSizeStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mOutName.c_str(), "recreate");
  mMassSpectrumFull = std::make_unique<TH1F>("V0", "V0 mass spectrum; MeV", 100, 1, -1); // auto-set axis ranges
  mMassSpectrumK0s = std::make_unique<TH1F>("K0s", "V0 mass spectrum; MeV", 100, 475, 525); // auto-set axis ranges
  mAvgClusSize = std::make_unique<TH1F>("avg_clus_size", "Average cluster size per track; pixels / cluster", 100, 1, -1); // auto-set axis ranges
  mCosPA = std::make_unique<TH1F>("CosPA", "cos(PA)", 100, 1, -1); // auto-set axis ranges
  mMassSpectrumFull->SetDirectory(nullptr);
  mMassSpectrumK0s->SetDirectory(nullptr);
  mAvgClusSize->SetDirectory(nullptr);
  mCosPA->SetDirectory(nullptr);
  LOGP(info, "TESTTEST init here");
}

void AvgClusSizeStudy::run(ProcessingContext& pc)
{
  LOGP(info, "TESTTEST starting run");
  auto geom = o2::its::GeometryTGeo::Instance();
  o2::globaltracking::RecoContainer recoData;
  LOGP(info, "TESTTEST collecting data");
  recoData.collectData(pc, *mDataRequest.get());
  LOGP(info, "TESTTEST collected data");
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void AvgClusSizeStudy::getClusterSizes(std::vector<int>& clusSizeVec, const gsl::span<const o2::itsmft::CompClusterExt> ITSclus, gsl::span<const unsigned char>::iterator& pattIt, const o2::itsmft::TopologyDictionary* mdict)
{
  for (unsigned int iClus{0}; iClus < ITSclus.size(); ++iClus) {
    auto& clus = ITSclus[iClus];
    auto pattID = clus.getPatternID();
    int npix;
    o2::itsmft::ClusterPattern patt;

    if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict->isGroup(pattID)) {
      patt.acquirePattern(pattIt);
      npix = patt.getNPixels();
    } else {

      npix = mdict->getNpixels(pattID);
      patt = mdict->getPattern(pattID);
    }
    clusSizeVec[iClus] = npix;
  }
  // LOG(info) << " Patt Npixel: " << pattVec[0].getNPixels();
}

void AvgClusSizeStudy::loadData(o2::globaltracking::RecoContainer& recoData)
{
  // mInputITStracks = recoData.getITSTracks();
  LOGP(info, "TESTTEST about to load data");
  mInputITSidxs = recoData.getITSTracksClusterRefs();

  auto compClus = recoData.getITSClusters();
  auto clusPatt = recoData.getITSClustersPatterns();
  auto pattIt = clusPatt.begin();
  mInputITSclusters.reserve(compClus.size());
  mInputClusterSizes.resize(compClus.size());
  o2::its::ioutils::convertCompactClusters(compClus, pattIt, mInputITSclusters, mDict);
  auto pattIt2 = clusPatt.begin();
  getClusterSizes(mInputClusterSizes, compClus, pattIt2, mDict);
}

void AvgClusSizeStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  loadData(recoData);
  // count total number of v0s, make sure it matches up at the end
  auto V0s = recoData.getV0s();
  // std::vector<int> trackClusterSizes;
  int totalSize = 0;
  double_t avgClusterSize;
  double_t mass;
  for (V0 v0 : V0s) {
    mCosPA->Fill(v0.getCosPA());
    mITStrack = recoData.getITSTrack(v0.getVertexID());
    auto firstClus = mITStrack.getFirstClusterEntry();
    auto ncl = mITStrack.getNumberOfClusters();
    for (int icl = 0; icl < ncl; icl++) {
      totalSize += mInputClusterSizes[mInputITSidxs[firstClus + icl]];
    }
    avgClusterSize = (double_t) totalSize / (double_t) ncl;
    LOGP(info, "[K0TEST] average mass is");
    
    mAvgClusSize->Fill(avgClusterSize);
    mass = std::sqrt(v0.calcMass2()) * 1000; // convert mass to MeV
    mMassSpectrumFull->Fill(mass);
    mMassSpectrumK0s->Fill(mass);
  }
  // if (V0s.size() == mInputITStracks.size()) {
  // };
  // double_t k0mass;
  // int id;
  // for (V0 k0 : K0s) {
  //   k0mass = std::sqrt(k0.calcMass2()) * 1000; // convert mass to MeV

  //   mMassSpectrumFull->Fill(k0mass);
  // }
}

void AvgClusSizeStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  LOGP(info, "TESTTEST begin update");
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}

void AvgClusSizeStudy::endOfStream(EndOfStreamContext& ec)
{
  LOGP(info, "TESTTEST eos");
  mDBGOut.reset();
  TFile fout(mOutName.c_str(), "update");
  fout.WriteTObject(mMassSpectrumFull.get());
  fout.WriteTObject(mMassSpectrumK0s.get());
  fout.WriteTObject(mAvgClusSize.get());
  fout.WriteTObject(mCosPA.get());
  LOGP(info, "Stored full mass spectrum histogram {} into {}", mMassSpectrumFull->GetName(), mOutName.c_str());
  LOGP(info, "Stored narrow mass spectrum histogram {} into {}", mMassSpectrumK0s->GetName(), mOutName.c_str());
  LOGP(info, "Stored cluster size histogram {} into {}", mAvgClusSize->GetName(), mOutName.c_str());
  LOGP(info, "Stored pointing angle histogram {} into {}", mCosPA->GetName(), mOutName.c_str());
  // can i store the mass data in a tree?? check itsoffsstudy.cxx 'itstof'
  
  
  TCanvas *c1 = new TCanvas();
  mMassSpectrumFull->Draw();
  c1->Print("massSpectrumFull.png");
  TCanvas *c2 = new TCanvas();
  mMassSpectrumK0s->Draw();
  c2->Print("massSpectrumK0s.png");
  TCanvas *c3 = new TCanvas();
  mAvgClusSize->Draw();
  c3->Print("clusSize.png");
  TCanvas *c4 = new TCanvas();
  mCosPA->Draw();
  c4->Print("cosPA.png");
  


  fout.Close();
}

void AvgClusSizeStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  LOGP(info, "TESTTEST ccdb finalise");
  {
  // if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
  //   return;
  // }
  o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOGP(info, "Cluster dictionary updated");
    setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
}

}

DataProcessorSpec getAvgClusSizeStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC)
// DataProcessorSpec getAvgClusSizeStudy(mask_t srcTracksMask, mask_t srcClusMask, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);
  dataRequest->requestClusters(srcClustersMask, useMC);
  dataRequest->requestSecondaryVertices(useMC);

  //TODO: is this even right???? need to figure out wtf these params mean
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  LOGP(info, "TESTTEST datareq done");
  return DataProcessorSpec{
    "its-study-AvgClusSize",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AvgClusSizeStudy>(dataRequest, ggRequest, useMC)},
    Options{}};
}
} // namespace study
} // namespace its
} // namespace o2