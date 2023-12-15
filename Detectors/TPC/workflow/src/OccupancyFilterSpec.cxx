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

#include <fmt/core.h>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <vector>
#include <gsl/span>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Headers/DataHeader.h"

#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCWorkflow/OccupancyFilterSpec.h"

using namespace o2::framework;
using namespace o2::header;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2::tpc
{

class OccupancyFilterDevice : public o2::framework::Task
{
 public:
  OccupancyFilterDevice() = default;

  void init(o2::framework::InitContext& ic) final
  {
    mOccupancyThreshold = ic.options().get<float>("occupancy-threshold");
    mAdcValueThreshold = ic.options().get<float>("adc-threshold");
    mMinimumTimeStamp = ic.options().get<long>("min-timestamp");
    mFilterAdcValue = ic.options().get<bool>("filter-adc");
    mFilterTimeStamp = ic.options().get<bool>("filter-timestamp");
    LOGP(debug, "got minimum timestamp {}", mMinimumTimeStamp);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    for (auto const& inputRef : InputRecordWalker(pc.inputs())) {
      auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(inputRef);
      if (sectorHeader == nullptr) {
        LOGP(error, "sector header missing on header stack for input on ", inputRef.spec->binding);
        continue;
      }

      const int sector = sectorHeader->sector();
      auto inDigitsO = pc.inputs().get<std::vector<o2::tpc::Digit>>(inputRef);
      LOGP(debug, "processing sector {} with {} input digits", sector, inDigitsO.size());

      std::vector<size_t> digitsPerTimeBin(int(128 * 445.5));

      for (const auto& digit : inDigitsO) {
        if ((digit.getChargeFloat() > mAdcValueThreshold) && (digit.getTimeStamp() > mMinimumTimeStamp)) {
          ++digitsPerTimeBin[digit.getTimeStamp()];
        }
      }

      bool isAboveThreshold{false};
      for (const auto& timeBinOccupancy : digitsPerTimeBin) {
        if (timeBinOccupancy > mOccupancyThreshold) {
          LOGP(info, "Sector {}, timeBinOccupancy {} > occupancy-threshold {}", sector, timeBinOccupancy, mOccupancyThreshold);
          isAboveThreshold = true;
          break;
        }
      }

      std::vector<o2::tpc::Digit> cpDigits;
      if (isAboveThreshold) {
        std::copy_if(inDigitsO.begin(), inDigitsO.end(), std::back_inserter(cpDigits),
                     [this](const auto& digit) {
                       return (!mFilterTimeStamp || (digit.getTimeStamp() > mMinimumTimeStamp)) && (!mFilterAdcValue || (digit.getChargeFloat() > mAdcValueThreshold));
                     });
      }
      snapshot(pc.outputs(), cpDigits, sector);

      ++mProcessedTFs;
      LOGP(info, "Number of processed time frames: {}", mProcessedTFs);
    }
  }

 private:
  float mOccupancyThreshold{50.f};
  float mAdcValueThreshold{0.f};
  long mMinimumTimeStamp{0LL};
  bool mFilterAdcValue{false};
  bool mFilterTimeStamp{false};
  uint32_t mProcessedTFs{0};

  //____________________________________________________________________________
  void snapshot(DataAllocator& output, std::vector<Digit>& digits, int sector)
  {
    o2::tpc::TPCSectorHeader header{sector};
    header.activeSectors = (0x1 << sector);
    output.snapshot(Output{gDataOriginTPC, "FILTERDIG", static_cast<SubSpecificationType>(sector), header}, digits);
  }
};

o2::framework::DataProcessorSpec getOccupancyFilterSpec()
{
  using device = o2::tpc::OccupancyFilterDevice;

  std::vector<InputSpec> inputs{
    InputSpec{"digits", gDataOriginTPC, "DIGITS", 0, Lifetime::Timeframe},
  };

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(gDataOriginTPC, "FILTERDIG", 0, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-occupancy-filter",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"occupancy-threshold", VariantType::Float, 50.f, {"threshold for occupancy in one time bin"}},
      {"adc-threshold", VariantType::Float, 0.f, {"threshold for adc value"}},
      {"min-timestamp", VariantType::Int64, 0LL, {"minimum time bin"}},
      {"filter-adc", VariantType::Bool, false, {"filter the data by the given adc threshold"}},
      {"filter-timestamp", VariantType::Bool, false, {"filter the data by the given minimum timestamp"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace o2::tpc
