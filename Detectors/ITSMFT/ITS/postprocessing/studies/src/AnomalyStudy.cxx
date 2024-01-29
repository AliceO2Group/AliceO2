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

#include "ITSStudies/AnomalyStudy.h"
#include "ITSStudies/ITSStudiesConfigParam.h"
#include "ITSBase/GeometryTGeo.h"

#include "Framework/Task.h"
#include "ITStracking/IOUtils.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

#include <TH2F.h>
#include <TCanvas.h>

namespace o2::its::study
{
using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;
using ITSCluster = o2::BaseCluster<float>;
class AnomalyStudy : public Task
{
  static constexpr int nChipStavesIB{9};
  static constexpr std::array<int, 3> nStavesIB{12, 16, 20};

 public:
  AnomalyStudy(std::shared_ptr<DataRequest> dr,
               std::shared_ptr<o2::base::GRPGeomRequest> gr,
               bool isMC) : mDataRequest{dr}, mGGCCDBRequest(gr), mUseMC(isMC) {}
  void init(InitContext& ic) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;

  // custom
  void prepareOutput();
  void updateTimeDependentParams(ProcessingContext& pc);
  void process(o2::globaltracking::RecoContainer& recoData);
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }

 private:
  bool mUseMC;
  int mTFCount{0};
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;

  // utils
  void getClusterPatterns(gsl::span<const o2::itsmft::CompClusterExt>&, gsl::span<const unsigned char>&, const o2::itsmft::TopologyDictionary&);
  std::vector<o2::itsmft::ClusterPattern> mPatterns;
  o2::its::GeometryTGeo* mGeom;
  o2::itsmft::ChipMappingITS mChipMapping;

  // Histos
  std::vector<std::unique_ptr<TH2F>> mTFvsPhiHist;
  std::vector<std::unique_ptr<TH2F>> mTFvsPhiClusSizeHist;
};

void AnomalyStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this param need to be queried only once
    initOnceDone = true;
    mGeom = o2::its::GeometryTGeo::Instance();
    mGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}

void AnomalyStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  prepareOutput();
  mTFvsPhiHist.resize(7);
  mTFvsPhiClusSizeHist.resize(7);
  auto nTF = o2::its::study::AnomalyStudyParamConfig::Instance().nTimeFramesOffset;
  for (unsigned int i = 0; i < 7; i++) {
    mTFvsPhiHist[i].reset(new TH2F(Form("tf_phi_occup_layer_%d", i), " ; #phi ; # TF; Counts", 150, -TMath::Pi(), TMath::Pi(), nTF, 0.5, nTF+0.5));
    mTFvsPhiClusSizeHist[i].reset(new TH2F(Form("tf_phi_clsize_layer_%d", i), "; #phi; # TF ; <Cluster Size>", 150, -TMath::Pi(), TMath::Pi(), nTF, 0.5, nTF+0.5));
  }
}

void AnomalyStudy::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void AnomalyStudy::endOfStream(EndOfStreamContext&)
{
  TFile* f = TFile::Open(o2::its::study::AnomalyStudyParamConfig::Instance().outFileName.c_str(), "recreate");
  // Iterate over all the histograms and write them to the file
  for (unsigned int i = 0; i < 7; i++) {
    mTFvsPhiClusSizeHist[i]->Divide(mTFvsPhiHist[i].get());
    mTFvsPhiHist[i]->Write();
    mTFvsPhiClusSizeHist[i]->Write();
  }

  f->Close();
}

void AnomalyStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
}

// custom
void AnomalyStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  mTFCount++;
  LOGP(info, "Processing TF: {}", mTFCount);
  // if (mTFCount > 50) {
  //   return;
  // }
  int rofCount = 0;
  auto clusRofRecords = recoData.getITSClustersROFRecords();
  auto compClus = recoData.getITSClusters();
  auto clusPatt = recoData.getITSClustersPatterns();

  mPatterns.clear();
  getClusterPatterns(compClus, clusPatt, *mDict);
  auto pattIt = clusPatt.begin();
  std::vector<ITSCluster> globalClusters;
  o2::its::ioutils::convertCompactClusters(compClus, pattIt, globalClusters, mDict);

  int lay, sta, ssta, mod, chipInMod;
  for (auto& rofRecord : clusRofRecords) {
    auto clustersInRof = rofRecord.getROFData(compClus);
    auto patternsInRof = rofRecord.getROFData(mPatterns);
    auto locClustersInRof = rofRecord.getROFData(globalClusters);
    for (unsigned int clusInd{0}; clusInd < clustersInRof.size(); clusInd++) {
      const auto& compClus = clustersInRof[clusInd];
      auto& locClus = locClustersInRof[clusInd];
      auto& clusPattern = patternsInRof[clusInd];
      auto gloC = locClus.getXYZGlo(*mGeom);
      mChipMapping.expandChipInfoHW(compClus.getChipID(), lay, sta, ssta, mod, chipInMod);
      mTFvsPhiHist[lay]->Fill(TMath::ATan2(gloC.Y(), gloC.X()), mTFCount);
      if(clusPattern.getNPixels() == 0) {
        LOGP(info, "Cluster with 0 pixels");
        LOGP(info, "LayerID: {}", lay);
      }
      mTFvsPhiClusSizeHist[lay]->Fill(TMath::ATan2(gloC.Y(), gloC.X()), mTFCount, clusPattern.getNPixels());
    }
  }
}

void AnomalyStudy::prepareOutput()
{
}

void AnomalyStudy::getClusterPatterns(gsl::span<const o2::itsmft::CompClusterExt>& ITSclus, gsl::span<const unsigned char>& ITSpatt, const o2::itsmft::TopologyDictionary& mdict)
{
  mPatterns.clear();
  mPatterns.resize(ITSclus.size());
  auto pattIt = ITSpatt.begin();

  for (unsigned int iClus{0}; iClus < ITSclus.size(); ++iClus) {
    auto& clus = ITSclus[iClus];

    auto pattID = clus.getPatternID();
    o2::itsmft::ClusterPattern patt;

    if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict.isGroup(pattID)) {
      patt.acquirePattern(pattIt);
    } else {
      patt = mdict.getPattern(pattID);
    }

    mPatterns.push_back(patt);
  }
}

// getter
DataProcessorSpec getAnomalyStudy(mask_t srcClustersMask, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestClusters(srcClustersMask, useMC);
  dataRequest->requestTracks(GTrackID::getSourcesMask(""), useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              false,                             // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  return DataProcessorSpec{
    "its-anomaly-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AnomalyStudy>(dataRequest, ggRequest, useMC)},
    Options{}};
}
} // namespace o2::its::study