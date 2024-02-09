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
#include <TF1.h>
#include <TCanvas.h>
#include <TStopwatch.h>

namespace o2::its::study
{
using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;
using ITSCluster = o2::BaseCluster<float>;
class AnomalyStudy : public Task
{
  static constexpr int nChipStavesIB{9};

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
  TStopwatch mStopwatch;
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
  std::vector<std::unique_ptr<TH2F>> mTFvsPhiHist;
  std::vector<std::unique_ptr<TH2F>> mTFvsPhiClusSizeHist;
  std::vector<std::unique_ptr<TH2F>> mROFvsPhiHist;
  std::vector<std::unique_ptr<TH2F>> mROFvsPhiClusSizeHist;
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

  auto nLayProc = o2::its::study::AnomalyStudyParamConfig::Instance().nLayersToProcess;
  auto nTF = o2::its::study::AnomalyStudyParamConfig::Instance().nTimeFramesOffset;
  auto nROF = o2::its::study::AnomalyStudyParamConfig::Instance().nRofTimeFrames;
  auto doROFAnalysis = o2::its::study::AnomalyStudyParamConfig::Instance().doROFAnalysis;

  mTFvsPhiHist.resize(nLayProc);
  mTFvsPhiClusSizeHist.resize(nLayProc);
  if (doROFAnalysis) {
    mROFvsPhiHist.resize(nLayProc);
    mROFvsPhiClusSizeHist.resize(nLayProc);
  }
  for (unsigned int i = 0; i < nLayProc; i++) {
    int phiBins = o2::its::study::AnomalyStudyParamConfig::Instance().nPhiBinsMultiplier * mNumberOfStaves[i];
    mTFvsPhiHist[i].reset(new TH2F(Form("tf_phi_occup_layer_%d", i), Form("Occupancy layer %d ; #phi ; # TF; Counts", i), phiBins, -TMath::Pi(), TMath::Pi(), nTF, 0.5, nTF + 0.5));
    mTFvsPhiClusSizeHist[i].reset(new TH2F(Form("tf_phi_clsize_layer_%d", i), Form("Cluster size layer %d ; #phi; # TF ; #lt Cluster Size #gt", i), phiBins, -TMath::Pi(), TMath::Pi(), nTF, 0.5, nTF + 0.5));
    if (doROFAnalysis) {
      mROFvsPhiHist[i].reset(new TH2F(Form("rof_phi_occup_layer_%d", i), Form("Occupancy layer %d ; #phi; # ROF; Counts", i), phiBins, -TMath::Pi(), TMath::Pi(), nROF * nTF, 0.5, nROF * nTF + 0.5));
      mROFvsPhiClusSizeHist[i].reset(new TH2F(Form("rof_phi_clsize_layer_%d", i), Form("Cluster size layer %d; #phi; # ROF; #lt Cluster Size #gt", i), phiBins, -TMath::Pi(), TMath::Pi(), nROF * nTF, 0.5, nROF * nTF + 0.5));
    }
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
  auto nLayProc = o2::its::study::AnomalyStudyParamConfig::Instance().nLayersToProcess;
  auto doROFAnalysis = o2::its::study::AnomalyStudyParamConfig::Instance().doROFAnalysis;

  // Iterate over all the histograms and compute the averages
  for (unsigned int i = 0; i < nLayProc; i++) {
    mTFvsPhiClusSizeHist[i]->Divide(mTFvsPhiHist[i].get());
    if (doROFAnalysis) {
      mROFvsPhiClusSizeHist[i]->Divide(mROFvsPhiHist[i].get());
    }
  }

  // Fit slices along x of the 2D histograms
  for (unsigned int iLayer = 0; iLayer < nLayProc; ++iLayer) {
    int phiBins = o2::its::study::AnomalyStudyParamConfig::Instance().nPhiBinsMultiplier * mNumberOfStaves[iLayer];
    TObjArray aSlices;
    auto* f1 = new TF1(Form("f1_%d", iLayer), "pol0", -TMath::Pi(), TMath::Pi());
    auto* hPValue = new TH1F(Form("pValue_%d", iLayer), Form("pValue_%d", iLayer), mTFvsPhiClusSizeHist[iLayer]->GetNbinsY(), 0.5, mTFvsPhiClusSizeHist[iLayer]->GetNbinsY() + 0.5);
    mTFvsPhiClusSizeHist[iLayer]->FitSlicesX(f1, 0, -1, 0, "QNR", &aSlices);
    auto* hChi2 = (TH1D*)aSlices.At(1);
    for (auto iTF{0}; iTF < hChi2->GetEntries(); ++iTF) {
      auto pValue = TMath::Prob(hChi2->GetBinContent(iTF + 1) * (phiBins - 1), phiBins - 1);
      // LOGP(info, "Layer: {} TF: {} Chi2: {} Pvalue: {}", iLayer, iTF, hChi2->GetBinContent(iTF + 1), pValue);
      hPValue->SetBinContent(iTF + 1, pValue);
    }
    hPValue->Write();
    // Save slices to file
    for (unsigned int j = 0; j < aSlices.GetEntries(); ++j) {
      auto h = (TH1D*)aSlices.At(j);
      h->SetMinimum(0);
      h->Write();
    }

    // Do the same for ROFs
    if (doROFAnalysis) {
      TObjArray aSlicesROF;
      auto f1ROF = new TF1(Form("f1ROF_%d", iLayer), "pol0", -TMath::Pi(), TMath::Pi());
      mROFvsPhiClusSizeHist[iLayer]->FitSlicesX(f1ROF, 0, -1, 0, "QNR", &aSlicesROF);
      // Save slices to file
      for (unsigned int j = 0; j < aSlicesROF.GetEntries(); ++j) {
        auto h = (TH1D*)aSlicesROF.At(j);
        h->SetMinimum(0);
        h->Write();
      }
    }
  }

  for (unsigned int i = 0; i < nLayProc; i++) {
    mTFvsPhiHist[i]->Write();
    mTFvsPhiClusSizeHist[i]->Write();
    if (doROFAnalysis) {
      mROFvsPhiHist[i]->Write();
      mROFvsPhiClusSizeHist[i]->Write();
    }
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
  mStopwatch.Start();
  mTFCount++;
  auto nROF = o2::its::study::AnomalyStudyParamConfig::Instance().nRofTimeFrames;
  auto nLayProc = o2::its::study::AnomalyStudyParamConfig::Instance().nLayersToProcess;
  auto doROFAnalysis = o2::its::study::AnomalyStudyParamConfig::Instance().doROFAnalysis;
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
      auto& clusPattern = patternsInRof[clusInd];
      auto gloC = locClus.getXYZGlo(*mGeom);
      mChipMapping.expandChipInfoHW(compClus.getChipID(), lay, sta, ssta, mod, chipInMod);
      if (lay >= nLayProc) {
        continue;
      }
      float phi = TMath::ATan2(gloC.Y(), gloC.X());
      mTFvsPhiHist[lay]->Fill(phi, mTFCount);
      mTFvsPhiClusSizeHist[lay]->Fill(phi, mTFCount, clusPattern.getNPixels());
      if (doROFAnalysis) {
        mROFvsPhiHist[lay]->Fill(phi, (mTFCount - 1) * nROF + rofCount);
        mROFvsPhiClusSizeHist[lay]->Fill(phi, (mTFCount - 1) * nROF + rofCount, clusPattern.getNPixels());
      }
    }
    ++rofCount;
  }
  mStopwatch.Stop();
  LOGP(info, "Processed TF: {} in {} s", mTFCount, mStopwatch.RealTime());
}

void AnomalyStudy::prepareOutput()
{
}

void AnomalyStudy::getClusterPatterns(gsl::span<const o2::itsmft::CompClusterExt>& ITSclus, gsl::span<const unsigned char>& ITSpatt, const o2::itsmft::TopologyDictionary& mdict)
{
  mPatterns.clear();
  mPatterns.reserve(ITSclus.size());
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