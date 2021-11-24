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

#include <iterator>
#include <memory>
#include <vector>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Headers/DataHeader.h"

#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCWorkflow/KryptonRawFilterSpec.h"

using namespace o2::framework;
using namespace o2::header;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace tpc
{

class KrRawFilterDevice : public o2::framework::Task
{
 public:
  KrRawFilterDevice() = default;

  void init(o2::framework::InitContext& ic) final
  {
    mThreshold = ic.options().get<float>("threshold");
    mTimeBinsBefore = ic.options().get<int>("time-bins-before");
    mTimeBinsAfter = ic.options().get<int>("time-bins-after");
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
      auto inDigitsO = pc.inputs().get<gsl::span<o2::tpc::Digit>>(inputRef);
      std::vector<Digit> inDigits(inDigitsO.begin(), inDigitsO.end());

      // sort pad-by-pad time bin increasing
      std::sort(inDigits.begin(), inDigits.end(), [](const auto& a, const auto& b) {
        if (a.getRow() < b.getRow()) {
          return true;
        }
        if (a.getRow() == b.getRow()) {
          if (a.getPad() < b.getPad()) {
            return true;
          } else if (a.getPad() == b.getPad()) {
            return a.getTimeStamp() < b.getTimeStamp();
          }
        }
        return false;
      });

      std::vector<Digit> digits;

      //
      //filter digits
      int lastPad = -1;
      int lastRow = -1;
      float maxCharge = 0;
      size_t posFirstTimeBin = 0;
      size_t posLastTimeBin = 0;
      size_t posMaxDigit = 0;

      for (size_t iDigit = 0; iDigit < inDigits.size(); ++iDigit) {
        const auto& digit = inDigits[iDigit];
        const auto pad = digit.getPad();
        const auto row = digit.getRow();
        const auto charge = digit.getChargeFloat();

        if (pad != lastPad || row != lastRow) {
          // write out last pad
          if (posMaxDigit > 0) {
            if (posMaxDigit - mTimeBinsBefore > posFirstTimeBin && posMaxDigit + mTimeBinsAfter < posLastTimeBin) {
              std::copy(&inDigits[posMaxDigit - mTimeBinsBefore], &inDigits[posMaxDigit + mTimeBinsAfter], std::back_inserter(digits));
            }
          }

          posFirstTimeBin = iDigit;
          posMaxDigit = 0;
          maxCharge = 0;
        }

        // center around max charge
        if (charge > mThreshold) {
          if (charge > maxCharge) {
            maxCharge = charge;
            posMaxDigit = iDigit;
          }
        }

        lastPad = pad;
        lastRow = row;
        posLastTimeBin = iDigit;
      }

      // copy also for last processed pad
      if (posMaxDigit > 0) {
        if (posMaxDigit - mTimeBinsBefore > posFirstTimeBin && posMaxDigit + mTimeBinsAfter < posLastTimeBin) {
          std::copy(&inDigits[posMaxDigit - mTimeBinsBefore], &inDigits[posMaxDigit + mTimeBinsAfter], std::back_inserter(digits));
        }
      }

      snapshot(pc.outputs(), digits, sector);

      LOGP(info, "processed sector {} with {} filtered digits", sector, digits.size());
    }

    ++mProcessedTFs;
    LOGP(info, "Number of processed time frames: {}", mProcessedTFs);
  }

 private:
  float mThreshold{50.f};
  int mTimeBinsBefore{10};
  int mTimeBinsAfter{100};
  uint32_t mProcessedTFs{0};

  //____________________________________________________________________________
  void snapshot(DataAllocator& output, std::vector<Digit>& digits, int sector)
  {
    o2::tpc::TPCSectorHeader header{sector};
    header.activeSectors = (0x1 << sector);
    output.snapshot(Output{gDataOriginTPC, "FILTERDIG", static_cast<SubSpecificationType>(sector), Lifetime::Timeframe, header}, digits);
  }
};

o2::framework::DataProcessorSpec getKryptonRawFilterSpec()
{
  using device = o2::tpc::KrRawFilterDevice;

  std::vector<InputSpec> inputs{
    InputSpec{"digits", gDataOriginTPC, "DIGITS", 0, Lifetime::Timeframe},
  };

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(gDataOriginTPC, "FILTERDIG", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-krypton-raw-filter",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"threshold", VariantType::Float, 50.f, {"threshold above which time sequences will be written out"}},
      {"time-bins-before", VariantType::Int, 10, {"time bins before trigger digit to be written"}},
      {"time-bins-after", VariantType::Int, 100, {"time bins after trigger digit to be written"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace tpc
} // namespace o2
