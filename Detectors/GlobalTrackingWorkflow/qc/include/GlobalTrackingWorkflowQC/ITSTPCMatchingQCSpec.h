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

#ifndef O2_ITSTPC_MATCHING_QC_DEVICE_H
#define O2_ITSTPC_MATCHING_QC_DEVICE_H

/// @file   ITSTPCMatchingQCDevice.h
/// @brief  Device to perform ITSTPC matching QC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "GLOQC/MatchITSTPCQC.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{
class ITSTPCMatchingQCDevice : public Task
{
 public:
  ITSTPCMatchingQCDevice(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> req, bool useMC, bool doK0QC, std::string trkSources) : mDataRequest(dr), mCCDBRequest(req), mUseMC(useMC), mDoK0QC(doK0QC), mTrkSources(trkSources){};
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void sendOutput(DataAllocator& output);
  std::unique_ptr<o2::gloqc::MatchITSTPCQC> mMatchITSTPCQC;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  bool mUseMC = true;
  bool mDoK0QC = true;
  std::string mTrkSources = "ITS,TPC,ITS-TPC";
};

} // namespace globaltracking

namespace framework
{
DataProcessorSpec getITSTPCMatchingQCDevice(bool useMC, bool doK0QC, std::string trkSources);

} // namespace framework
} // namespace o2

#endif
