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
#include "TPCBase/Mapper.h"
#include "TPCBase/Sector.h"
#include "TPCWorkflow/KryptonRawFilterSpec.h"

using namespace o2::framework;
using namespace o2::header;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2::tpc
{

class KrRawFilterDevice : public o2::framework::Task
{
 public:
  KrRawFilterDevice(CalPad* noise) : mNoise(noise) {}

  void init(o2::framework::InitContext& ic) final
  {
    mThresholdMax = ic.options().get<float>("threshold-max");
    mThresholdAbs = ic.options().get<float>("threshold-abs");
    mThresholdSigma = ic.options().get<float>("threshold-sigma");
    mTimeBinsBefore = ic.options().get<int>("time-bins-before");
    mTimeBinsAfter = ic.options().get<int>("time-bins-after");
    mMaxTimeBins = ic.options().get<int>("max-time-bins");
    mSkipIROCTSensors = ic.options().get<bool>("skip-iroc-tsen");
    LOGP(info, "threshold-max: {}, threshold-abs: {}, threshold-sigma: {}, time-bins-before: {}, time-bins-after: {}, max-time-bins: {}, skip-iroc-tsen: {}", mThresholdMax, mThresholdAbs, mThresholdSigma, mTimeBinsBefore, mTimeBinsAfter, mMaxTimeBins, mSkipIROCTSensors);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto& mapper = Mapper::instance();
    const size_t MAXROWS = 152;
    const size_t MAXPADS = 138;

    for (auto const& inputRef : InputRecordWalker(pc.inputs())) {
      auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(inputRef);
      if (sectorHeader == nullptr) {
        LOGP(error, "sector header missing on header stack for input on ", inputRef.spec->binding);
        continue;
      }

      const int sector = sectorHeader->sector();
      auto inDigitsO = pc.inputs().get<gsl::span<o2::tpc::Digit>>(inputRef);
      LOGP(debug, "processing sector {} with {} input digits", sector, inDigitsO.size());

      // ===| flatten digits for simple access and register interesting digit positions  |===
      // flat digit structure
      std::vector<float> inDigits(MAXROWS * MAXPADS * mMaxTimeBins);

      // counter how often a pad triggered to filter extremely noisy or struck pads
      std::vector<size_t> nTriggered(MAXROWS * MAXPADS);

      struct DigitInfo {
        int row{};
        int pad{};
        int time{};
      };
      std::vector<DigitInfo> maxDigits;

      for (const auto& digit : inDigitsO) {
        const auto time = digit.getTimeStamp();
        if (time >= mMaxTimeBins) {
          continue;
        }

        const auto pad = digit.getPad();
        const auto row = digit.getRow();
        const auto charge = digit.getChargeFloat();

        const size_t offset = row * MAXPADS * mMaxTimeBins + pad * mMaxTimeBins;
        inDigits[offset + time] = charge;

        const auto noiseThreshold = mNoise ? mThresholdSigma * mNoise->getValue(Sector(sector), row, pad) : 0.f;
        if ((charge > mThresholdMax) && (charge > noiseThreshold)) {
          if (!(mSkipIROCTSensors && (row == 47) && (pad > 40))) {
            if ((time > mTimeBinsBefore) && (time < (mMaxTimeBins - mTimeBinsAfter))) {
              maxDigits.emplace_back(DigitInfo{row, pad, time});
              ++nTriggered[row * MAXPADS + pad];
            }
          }
        }
      }

      // ===| find local maxima |===

      // simple local maximum detection in time and pad direction
      auto isLocalMaximum = [&inDigits, this](const int row, const int pad, const int time) {
        const size_t offset = row * MAXPADS * mMaxTimeBins + pad * mMaxTimeBins + time;
        const auto testCharge = inDigits[offset];

        // look in time direction
        if (!(testCharge > inDigits[offset + 1])) {
          return false;
        }
        if (testCharge < inDigits[offset - 1]) {
          return false;
        }

        // look in pad direction
        if (!(testCharge > inDigits[offset + mMaxTimeBins])) {
          return false;
        }
        if (testCharge < inDigits[offset - mMaxTimeBins]) {
          return false;
        }

        // check diagonals
        if (!(testCharge > inDigits[offset + mMaxTimeBins + 1])) {
          return false;
        }
        if (!(testCharge > inDigits[offset + mMaxTimeBins - 1])) {
          return false;
        }
        if (testCharge < inDigits[offset - mMaxTimeBins + 1]) {
          return false;
        }
        if (testCharge < inDigits[offset - mMaxTimeBins - 1]) {
          return false;
        }

        return true;
      };

      // digit filter
      std::vector<Digit> digits;

      // fill digits and return true if the neigbouring pad should be filtered as well
      auto fillDigits = [&digits, &inDigits, &sector, this](const size_t row, const size_t pad, const size_t time) {
        int cru = Mapper::REGION[row] + sector * Mapper::NREGIONS;
        const size_t offset = row * MAXPADS * mMaxTimeBins + pad * mMaxTimeBins;

        const auto chargeMax = inDigits[offset + time];

        for (int iTime = time - mTimeBinsBefore; iTime < time + mTimeBinsAfter; ++iTime) {
          const auto charge = inDigits[offset + iTime];
          if (charge < -999.f) { // avoid double copy
            continue;
          }
          digits.emplace_back(Digit(cru, charge, row, pad, iTime));
          inDigits[offset + iTime] = -1000.f;
        }

        const auto noiseThreshold = mNoise ? mThresholdSigma * mNoise->getValue(Sector(sector), row, pad) : 0.f;
        return (chargeMax > mThresholdAbs) && (chargeMax > noiseThreshold);
      };

      // filter digits
      for (const auto& maxDigit : maxDigits) {
        const auto row = maxDigit.row;
        const auto pad = maxDigit.pad;
        // skip extremely noisy pads
        if (nTriggered[row * MAXPADS + pad] > mMaxTimeBins / 4) {
          continue;
        }
        const auto time = maxDigit.time;
        const int nPads = mapper.getNumberOfPadsInRowSector(row);
        // skip edge pads
        if ((pad == 0) || (pad == nPads - 1)) {
          continue;
        }

        if (isLocalMaximum(row, pad, time)) {
          auto padRef = pad;
          while ((padRef >= 0) && fillDigits(row, padRef, time)) {
            --padRef;
          }
          padRef = pad + 1;
          while ((padRef < nPads) && fillDigits(row, padRef, time)) {
            ++padRef;
          }
        }
      }

      snapshot(pc.outputs(), digits, sector);

      LOGP(info, "processed sector {} with {} input and {} filtered digits", sector, inDigitsO.size(), digits.size());
    }

    ++mProcessedTFs;
    LOGP(info, "Number of processed time frames: {}", mProcessedTFs);
  }

 private:
  float mThresholdMax{50.f};
  float mThresholdAbs{20.f};
  float mThresholdSigma{10.f};
  int mTimeBinsBefore{10};
  int mTimeBinsAfter{100};
  size_t mMaxTimeBins{450};
  uint32_t mProcessedTFs{0};
  bool mSkipIROCTSensors{true};
  CalPad* mNoise{nullptr};

  //____________________________________________________________________________
  void snapshot(DataAllocator& output, std::vector<Digit>& digits, int sector)
  {
    o2::tpc::TPCSectorHeader header{sector};
    header.activeSectors = (0x1 << sector);
    output.snapshot(Output{gDataOriginTPC, "FILTERDIG", static_cast<SubSpecificationType>(sector), header}, digits);
  }
};

o2::framework::DataProcessorSpec getKryptonRawFilterSpec(CalPad* noise)
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
    AlgorithmSpec{adaptFromTask<device>(noise)},
    Options{
      {"threshold-max", VariantType::Float, 50.f, {"threshold in absolute ADC counts for the maximum digit"}},
      {"threshold-abs", VariantType::Float, 20.f, {"threshold in absolute ADC counts above which time sequences will be written out"}},
      {"threshold-sigma", VariantType::Float, 10.f, {"threshold in sigma noise above which time sequences will be written out"}},
      {"time-bins-before", VariantType::Int, 10, {"time bins before trigger digit to be written"}},
      {"time-bins-after", VariantType::Int, 100, {"time bins after trigger digit to be written"}},
      {"max-time-bins", VariantType::Int, 450, {"maximum number of time bins to process"}},
      {"skip-iroc-tsen", VariantType::Bool, true, {"skip IROC T-Sensor pads in maxima detection"}},
    } // end Options
  };  // end DataProcessorSpec
}
} // namespace o2::tpc
