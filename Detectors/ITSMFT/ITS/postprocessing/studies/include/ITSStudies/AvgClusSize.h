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


#include <TH1F.h>
#include <THStack.h>


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

class AvgClusSizeStudy : public Task
{
 public:
  AvgClusSizeStudy(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool isMC) : mDataRequest{dr}, mGGCCDBRequest(gr), mUseMC(isMC){};
  ~AvgClusSizeStudy() = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void process(o2::globaltracking::RecoContainer&);
  void loadData(o2::globaltracking::RecoContainer&);
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }
  void getClusterSizes(std::vector<int>&, const gsl::span<const o2::itsmft::CompClusterExt>, gsl::span<const unsigned char>::iterator&, const o2::itsmft::TopologyDictionary*);
  void saveHistograms();
  void formatHistograms();
  void plotHistograms();
  void findEtaBin(double eta, double clusSize, int i);
  void fitMassSpectrum();

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  bool mUseMC;

  // Data
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  std::vector<int> mInputClusterSizes;
  gsl::span<const int> mInputITSidxs;
  gsl::span<const TrackITS> mInputITStracks;
  std::vector<ITSCluster> mInputITSclusters;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;

  // Output
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
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
  std::vector<double> etaBinEdges;
  std::unique_ptr<TH1F> mCosPA{};
  std::unique_ptr<TH1F> mR{};
  std::unique_ptr<TH1F> mDCA{};
  std::unique_ptr<TH1F> mEtaNC{};
  std::unique_ptr<TH1F> mEtaC{};
  int globalNClusters;
  int globalNPixels;
  double etaMin = -1.5;
  double etaMax = 1.5;
  int etaNBins = 5;


  const std::string mOutName{"massSpectrum.root"};
};






o2::framework::DataProcessorSpec getAvgClusSizeStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC);
} // namespace study
} // namespace its
} // namespace o2

#endif