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

#include <TH2D.h>
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
  const int mNumberOfStaves[7] = {12, 16, 20, 24, 30, 42, 48};
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;

  // utils
  void getClusterPatterns(gsl::span<const o2::itsmft::CompClusterExt>&, gsl::span<const unsigned char>&, const o2::itsmft::TopologyDictionary&);
  std::vector<o2::itsmft::ClusterPattern> mPatterns;
  o2::its::GeometryTGeo* mGeom;
  o2::itsmft::ChipMappingITS mChipMapping;

  // Histos
  std::vector<std::unique_ptr<TH2D>> mTFvsPhiHist;
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
  int count{0};
  mTFvsPhiHist.resize(7);
  for (auto& hist : mTFvsPhiHist) {
    hist.reset(new TH2D(Form("tf_phi_layer_%d", count), Form("tf_phi_layer_%d", count), mNumberOfStaves[count] * 10, -TMath::Pi(), TMath::Pi(), 100, 0.5, 100.5));
    count++;
  }
  LOGP(info, "Initialized {} TFvsPhi histos", mTFvsPhiHist.size());
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
  for (auto& hist : mTFvsPhiHist) {
    hist->Write();
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
  int rofCount = 0;
  auto clusRofRecords = recoData.getITSClustersROFRecords();
  auto compClus = recoData.getITSClusters();
  auto clusPatt = recoData.getITSClustersPatterns();
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
      mChipMapping.expandChipInfoHW(compClus.getChipID(), lay, sta, ssta, mod, chipInMod);

      // LOG(info) << "Cluster in layer " << lay << " stave " << sta << " sub-stave " << ssta << " module " << mod << " chip " << chipInMod;
      // LOG(info) << "Cluster in layer " << locClus.getLayer() << " stave " << locClus.getStave() << " sub-stave " << locClus.getSubStave() << " module " << locClus.getModule() << " chip " << locClus.getChipID();
      // LOGP(info, "Filling {} layer with phi {} and count {}", lay, TMath::ATan2(locClus.getY(), locClus.getX()), mTFCount);
      auto gloC = locClus.getXYZGlo(*mGeom);
      mTFvsPhiHist[lay]->Fill(TMath::ATan2(gloC.Y(), gloC.X()), mTFCount);
      // if (!lay) { // Inner barrel
      //   auto col = clus.getCol();
      //   auto row = clus.getRow();
      //   int ic = 0, ir = 0;

      //   auto colSpan = patternsInRof[clusInd].getColumnSpan();
      //   auto rowSpan = patternsInRof[clusInd].getRowSpan();
      //   auto nBits = rowSpan * colSpan;

      //   for (int i = 2; i < patternsInRof[clusInd].getUsedBytes() + 2; i++) {
      //     unsigned char tempChar = patternsInRof[clusInd].getByte(i);
      //     int s = 128; // 0b10000000
      //     while (s > 0) {
      //       if ((tempChar & s) != 0) // checking active pixels
      //       {
      //         mLayerChipHists[lay][sta * nChipStavesIB + chipInMod]->Fill(col + ic, row + ir);
      //       }
      //       ic++;
      //       s >>= 1;
      //       if ((ir + 1) * ic == nBits) {
      //         break;
      //       }
      //       if (ic == colSpan) {
      //         ic = 0;
      //         ir++;
      //       }
      //       if ((ir + 1) * ic == nBits) {
      //         break;
      //       }
      //     }
      //   }
      // }
    }
  }
}

void AnomalyStudy::prepareOutput()
{
  // auto& params = o2::its::study::ITSClusDistributionParamConfig::Instance();
  // mNClusDistHist = std::make_unique<TH1F>("nClusDist", "Cluster distribution", 300, 0, 200e3);
  // mLayerChipHists.resize(3); // only inner barrel atm
  // int lCount{0};
  // for (auto& lHists : mLayerChipHists) {
  //   lHists.resize(nChipStavesIB * nStavesIB[lCount]);
  //   int cID{0};
  //   for (auto& cHist : lHists) {
  //     cHist = std::make_unique<TH2D>(Form("l%d_s%d_c%d", lCount, cID / nChipStavesIB, cID % nChipStavesIB),
  //                                    Form("l%d_s%d_c%d", lCount, cID / nChipStavesIB, cID % nChipStavesIB),
  //                                    256, -0.5, 1023.5, 128, -0.5, 511.5);
  //     cID++;
  //   }
  //   lCount++;
  // }
}

void AnomalyStudy::getClusterPatterns(gsl::span<const o2::itsmft::CompClusterExt>& ITSclus, gsl::span<const unsigned char>& ITSpatt, const o2::itsmft::TopologyDictionary& mdict)
{
  mPatterns.reserve(ITSclus.size());
  auto pattIt = ITSpatt.begin();

  for (unsigned int iClus{0}; iClus < ITSclus.size(); ++iClus) {
    auto& clus = ITSclus[iClus];

    auto pattID = clus.getPatternID();
    int npix;
    o2::itsmft::ClusterPattern patt;

    if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict.isGroup(pattID)) {
      patt.acquirePattern(pattIt);
      npix = patt.getNPixels();
    } else {
      npix = mdict.getNpixels(pattID);
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