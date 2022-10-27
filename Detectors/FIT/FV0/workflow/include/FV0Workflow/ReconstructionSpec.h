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

/// @file   ReconstructionSpec.h

#ifndef O2_FV0RECONSTRUCTORDPL_H
#define O2_FV0RECONSTRUCTORDPL_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "FV0Reconstruction/BaseRecoTask.h"
#include "DataFormatsFV0/RecPoints.h"
#include "FV0Base/Constants.h"
#include "TStopwatch.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace fv0
{

class ReconstructionDPL : public Task
{
  static constexpr int NCHANNELS = o2::fv0::Constants::nFv0Channels;

 public:
  ReconstructionDPL(bool useMC, const std::string ccdbpath) : mUseMC(useMC), mCCDBpath(ccdbpath) {}
  ~ReconstructionDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  bool mUseMC = false;
  bool mUpdateCCDB = true;
  const std::string mCCDBpath = o2::base::NameConf::getCCDBServer();
  std::vector<o2::fv0::RecPoints> mRecPoints;
  std::vector<o2::fv0::ChannelDataFloat> mRecChData;
  o2::fv0::BaseRecoTask mReco;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFV0;
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getReconstructionSpec(bool useMC = false, const std::string ccdbpath = "https://alice-ccdb.cern.ch");

} // namespace fv0
} // namespace o2

#endif /* O2_FV0RECONSTRUCTORDPL_H */
