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

/// @file   ClustererSpec.h

#ifndef O2_ITSMFT_CLUSTERERDPL_H_
#define O2_ITSMFT_CLUSTERERDPL_H_

#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ITSMFTReconstruction/Clusterer.h"

using namespace o2::framework;

namespace o2::itsmft
{

template <int N>
class ClustererDPL : public Task
{
  static constexpr o2::detectors::DetID ID{N == o2::detectors::DetID::ITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT};
  static constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};

 public:
  ClustererDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC, bool doStag);
  ~ClustererDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);

  std::string mDetName;
  bool mUseMC = true;
  bool mUseClusterDictionary = true;
  int mNThreads = 1;
  std::unique_ptr<o2::itsmft::Clusterer> mClusterer = nullptr;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mDoStaggering = false;
  int mLayers = 1;
  std::vector<InputSpec> mFilter;
};

framework::DataProcessorSpec getITSClustererSpec(bool useMC, bool doStag);
framework::DataProcessorSpec getMFTClustererSpec(bool useMC, bool doStag);

} // namespace o2::itsmft

#endif /* O2_MFT_CLUSTERERDPL */
