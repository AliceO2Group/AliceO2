// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/RawAggregatorSpec.cxx
/// \brief  Data processor spec for MID raw data aggregator device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   26 February 2020

#include "MIDWorkflow/RawAggregatorSpec.h"

#include <chrono>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include <Framework/Logger.h>
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDRaw/DecodedDataAggregator.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class RawAggregatorDeviceDPL
{
 public:
  void init(of::InitContext& ic)
  {
    auto stop = [this]() {
      LOG(INFO) << "Capacities: ROFRecords: " << mAggregator.getROFRecords().capacity() << "  Data: " << mAggregator.getData().capacity();
      double scaleFactor = 1.e6 / mNROFs;
      LOG(INFO) << "Processing time / " << mNROFs << " ROFs: full: " << mTimer.count() * scaleFactor << "  aggregating: " << mTimerAlgo.count() * scaleFactor << " us";
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);
  }

  void run(of::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();

    auto msg = pc.inputs().get("mid_decoded");
    auto data = of::DataRefUtils::as<const ROBoard>(msg);

    auto msgROF = pc.inputs().get("mid_decoded_rof");
    auto inROFRecords = of::DataRefUtils::as<const ROFRecord>(msgROF);

    auto tAlgoStart = std::chrono::high_resolution_clock::now();
    mAggregator.process(data, inROFRecords);
    mTimerAlgo += std::chrono::high_resolution_clock::now() - tAlgoStart;

    pc.outputs().snapshot(of::Output{"MID", "DATA", 0, of::Lifetime::Timeframe}, mAggregator.getData());
    pc.outputs().snapshot(of::Output{"MID", "DATAROF", 0, of::Lifetime::Timeframe}, mAggregator.getROFRecords());

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
    mNROFs += mAggregator.getROFRecords().size();
  }

 private:
  DecodedDataAggregator mAggregator{};
  std::chrono::duration<double> mTimer{0};     ///< full timer
  std::chrono::duration<double> mTimerAlgo{0}; ///< Algorithm timer
  unsigned int mNROFs{0};                      /// Total number of processed ROFs
};

framework::DataProcessorSpec getRawAggregatorSpec()
{
  std::vector<of::InputSpec> inputSpecs{of::InputSpec{"mid_decoded", header::gDataOriginMID, "DECODED"}, of::InputSpec{"mid_decoded_rof", header::gDataOriginMID, "DECODEDROF"}};
  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{header::gDataOriginMID, "DATA"}, of::OutputSpec{header::gDataOriginMID, "DATAROF"}};

  return of::DataProcessorSpec{
    "MIDRawAggregator",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::RawAggregatorDeviceDPL>()}};
}
} // namespace mid
} // namespace o2
