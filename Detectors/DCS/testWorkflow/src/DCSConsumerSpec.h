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

#ifndef O2_DCS_CONSUMER_H
#define O2_DCS_CONSUMER_H

/// @file   DCSConsumerSpec.h
/// @brief  Consumer of DPs coming from DCS server; it is just
/// to check that we receive and pack the data correctly

#include "DetectorsDCS/DataPointCompositeObject.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/DataSpecUtils.h"
#include <TStopwatch.h>

using namespace o2::framework;
namespace o2h = o2::header;

namespace o2
{
namespace dcs
{
class DCSConsumer : public o2::framework::Task
{

  using DPCOM = o2::dcs::DataPointCompositeObject;

 public:
  void init(o2::framework::InitContext& ic) final
  {
    mReportTiming = ic.options().get<bool>("report-timing");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    TStopwatch sw;
    uint64_t tfid;
    for (auto& input : pc.inputs()) {
      tfid = header::get<o2::framework::DataProcessingHeader*>(input.header)->startTime;
      LOG(debug) << "tfid = " << tfid;
      break; // we break because one input is enough to get the TF ID
    }

    LOG(debug) << "TF: " << tfid << " --> reading binary blob...";
    mTFs++;
    auto vect = pc.inputs().get<gsl::span<DPCOM>>("COMMONDPs");
    LOG(info) << "vector has " << vect.size() << " Data Points inside";
    sw.Stop();
    if (mReportTiming) {
      LOGP(info, "Timing CPU:{:.3e} Real:{:.3e} at slice {}", sw.CpuTime(), sw.RealTime(), pc.services().get<o2::framework::TimingInfo>().timeslice);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {

    LOG(info) << "Number of processed TFs = " << mTFs;
  }

 private:
  uint64_t mTFs = 0;
  bool mReportTiming = false;
};

} // namespace dcs

namespace framework
{

DataProcessorSpec getDCSConsumerSpec()
{
  return DataProcessorSpec{
    "dcs-consumer",
    Inputs{{"COMMONDPs", "DCS", "COMMON", 0, Lifetime::Timeframe}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<o2::dcs::DCSConsumer>()},
    Options{{"report-timing", VariantType::Bool, false, {"Report timing for every slice"}}}};
}

} // namespace framework
} // namespace o2

#endif
