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

#ifndef O2_AVGCLUSSIZE_STUDY_H
#define O2_AVGCLUSSIZE_STUDY_H

#include "Framework/DataProcessorSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ITStracking/IOUtils.h"
#include "DataFormatsITS/TrackITS.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "Framework/Task.h"
#include <Steer/MCKinematicsReader.h>

#include "ITSStudies/ITSStudiesConfigParam.h"

#include <TH1F.h>
#include <THStack.h>
#include <TTree.h>

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;
using ITSCluster = o2::BaseCluster<float>;
using mask_t = o2::dataformats::GlobalTrackID::mask_t;
using MCLabel = o2::MCCompLabel;

class AvgClusSizeStudy : public Task
{
 public:
  AvgClusSizeStudy(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool isMC) : mDataRequest{dr}, mGGCCDBRequest(gr), mUseMC(isMC){};
  ~AvgClusSizeStudy() = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }

 private:
  // Other functions
  void process(o2::globaltracking::RecoContainer&);
  void loadData(o2::globaltracking::RecoContainer&);

  // Helper functions
  void prepareOutput();
  void setStyle();
  void updateTimeDependentParams(ProcessingContext& pc);
  double getAverageClusterSize(o2::its::TrackITS);
  void getClusterSizes(std::vector<int>&, const gsl::span<const o2::itsmft::CompClusterExt>, gsl::span<const unsigned char>::iterator&, const o2::itsmft::TopologyDictionary*);
  void fitMassSpectrum();
  void saveHistograms();
  void plotHistograms();
  void fillEtaBin(double eta, double clusSize, int i);

  // Running options
  bool mUseMC;
  const o2::its::study::AvgClusSizeStudyParamConfig& mParams = o2::its::study::AvgClusSizeStudyParamConfig::Instance(); // NOTE: unsure if this is implemented in the "typical" way - it does work though

  // Data
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  std::vector<int> mInputClusterSizes;
  gsl::span<const int> mInputITSidxs;
  std::vector<o2::MCTrack> mMCTracks;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;

  // Output plots
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::unique_ptr<TTree> mOutputTree;

  std::unique_ptr<THStack> mMassSpectrumFull{};
  std::unique_ptr<TH1F> mMassSpectrumFullNC{};
  std::unique_ptr<TH1F> mMassSpectrumFullC{};
  std::unique_ptr<THStack> mMassSpectrumK0s{};
  std::unique_ptr<TH1F> mMassSpectrumK0sNC{};
  std::unique_ptr<TH1F> mMassSpectrumK0sC{};
  std::unique_ptr<THStack> mAvgClusSize{};
  std::unique_ptr<TH1F> mAvgClusSizeNC{};
  std::unique_ptr<TH1F> mAvgClusSizeC{};
  std::unique_ptr<THStack> mAvgClusSizeCEta{};
  std::vector<std::unique_ptr<TH1F>> mAvgClusSizeCEtaVec{};
  std::unique_ptr<THStack> mMCStackCosPA{};
  std::unique_ptr<THStack> mStackDCA{};
  std::unique_ptr<THStack> mStackR{};
  std::unique_ptr<THStack> mStackPVDCA{};
  std::unique_ptr<TH1F> mCosPA{};
  std::unique_ptr<TH1F> mMCCosPA_K0{};
  std::unique_ptr<TH1F> mMCCosPA_notK0{};
  std::unique_ptr<TH1F> mCosPA_trueK0{};
  std::unique_ptr<TH1F> mR{};
  std::unique_ptr<TH1F> mR_K0{};
  std::unique_ptr<TH1F> mR_notK0{};
  std::unique_ptr<TH1F> mR_trueK0{};
  std::unique_ptr<TH1F> mDCA{};
  std::unique_ptr<TH1F> mDCA_K0{};
  std::unique_ptr<TH1F> mDCA_notK0{};
  std::unique_ptr<TH1F> mDCA_trueK0{};
  std::unique_ptr<TH1F> mEtaNC{};
  std::unique_ptr<TH1F> mEtaC{};
  std::unique_ptr<TH1F> mMCMotherPDG{};
  std::unique_ptr<TH1F> mPVDCA_K0{};
  std::unique_ptr<TH1F> mPVDCA_notK0{};

  int globalNClusters = 0;
  int globalNPixels = 0;

  std::vector<double> mEtaBinUL; // upper edges for eta bins

  // Counters for K0s identification
  int nNotValid = 0;
  int nNullptrs = 0;
  int nPiPi = 0;
  int nIsPiPiNotK0s = 0;
  int nIsPiPiIsK0s = 0;
  int nIsNotPiPiIsK0s = 0;
  int nMotherIDMismatch = 0;
  int nEvIDMismatch = 0;
  int nK0s = 0;
  int nNotK0s = 0;
  int nPionsInEtaRange = 0;
  int nInvalidK0sMother = 0;

  const std::string mOutName{"o2standalone_cluster_size_study.root"};
  std::unique_ptr<o2::steer::MCKinematicsReader> mMCKinReader;
};

o2::framework::DataProcessorSpec getAvgClusSizeStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC);
} // namespace study
} // namespace its
} // namespace o2

#endif