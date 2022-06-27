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

/// @file GRPDCSDPsSpec.h

#ifndef O2_GRP_DCS_DPs_SPEC
#define O2_GRP_DCS_DPs_SPEC

#include <unistd.h>
#include "GRPCalibration/GRPDCSDPsProcessor.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

using namespace o2::framework;
using HighResClock = std::chrono::high_resolution_clock;

namespace o2
{
namespace grp
{
class GRPDCSDPsDataProcessor : public Task
{
 public:
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  void sendLHCIFDPsoutput(DataAllocator& output);
  void sendMagFieldDPsoutput(DataAllocator& output);
  void sendCollimatorsDPsoutput(DataAllocator& output);
  void sendEnvVarsDPsoutput(DataAllocator& output);

  bool mVerbose = false; // to enable verbose mode
  std::unique_ptr<GRPDCSDPsProcessor> mProcessor;
  HighResClock::time_point mTimer;
  int64_t mDPsUpdateInterval;
  bool mReportTiming = false;
  bool mLHCIFupdated = false;
};
} // namespace grp

namespace framework
{
DataProcessorSpec getGRPDCSDPsDataProcessorSpec();

} // namespace framework
} // namespace o2

#endif
