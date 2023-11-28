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

#ifndef O2_FT0RECONSTRUCTORDPL_H
#define O2_FT0RECONSTRUCTORDPL_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "FT0Reconstruction/CollisionTimeRecoTask.h"
#include "DataFormatsFT0/RecPoints.h"
#include "FT0Base/Geometry.h"
#include "TStopwatch.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

class ReconstructionDPL : public Task
{
  static constexpr int NCHANNELS = o2::ft0::Geometry::Nchannels;

 public:
  ReconstructionDPL(bool useMC, const std::string& ccdbpath, bool useTimeOffsetCalib, bool useSlewingCalib) : mUseMC(useMC), mCCDBpath(ccdbpath), mUseTimeOffsetCalib(useTimeOffsetCalib), mUseSlewingCalib(useSlewingCalib) {}
  ~ReconstructionDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  bool mUseMC = false;
  bool mUpdateCCDB = true;
  bool mUseTimeOffsetCalib = true;
  bool mUseSlewingCalib = true;
  const std::string mCCDBpath = o2::base::NameConf::getCCDBServer();
  std::vector<o2::ft0::RecPoints> mRecPoints;
  std::vector<o2::ft0::ChannelDataFloat> mRecChData;
  o2::ft0::CollisionTimeRecoTask mReco;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFT0;
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getReconstructionSpec(bool useMC = false, const std::string ccdbpath = "http://alice-ccdb.cern.ch", bool useTimeOffsetCalib = true, bool useSlewingCalib = true);

} // namespace ft0
} // namespace o2

#endif /* O2_FT0RECONSTRUCTORDPL_H */
