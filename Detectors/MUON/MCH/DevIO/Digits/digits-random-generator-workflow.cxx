// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DigitIOBaseTask.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "MCHMappingInterface/Segmentation.h"
#include "ProgOptions.h"
#include <array>
#include <chrono>
#include <cstring>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

/**
 * This executable generate random MCH digits and their associated ROFRecords.
 *
 */

using namespace o2::framework;

using namespace o2::mch;

namespace
{
std::array<int, 156> getDeIds()
{
  static std::array<int, 156> deids;
  static bool first = true;
  if (first) {
    int i{0};
    o2::mch::mapping::forEachDetectionElement([&](int deid) {
      deids[i++] = deid;
    });
    first = false;
  }
  return deids;
}

std::array<int, 156> getNofPads()
{
  static bool first = true;
  static std::array<int, 156> npads;
  if (first) {
    int i{0};
    auto deids = getDeIds();
    for (auto deid : deids) {
      o2::mch::mapping::Segmentation seg(deid);
      int16_t nofPads = static_cast<int16_t>(seg.nofPads());
      npads[i++] = nofPads;
    }
    first = false;
  }
  return npads;
}
} // namespace

constexpr const char* OPTNAME_NOF_ROFS_PER_TF = "nof-rofs-per-tf";
constexpr const char* OPTNAME_OCCUPANCY = "occupancy";
constexpr const char* OPTNAME_SEED = "seed";

class DigitGenerator : public o2::mch::io::DigitIOBaseTask
{
 public:
  void init(InitContext& ic)
  {
    io::DigitIOBaseTask::init(ic); // init common options
    mOccupancy = ic.options().get<float>(OPTNAME_OCCUPANCY);
    if (mOccupancy <= 0.0 || mOccupancy > 1.0) {
      throw std::invalid_argument("occupancy must be between >0 and <=1");
    }
    mNofRofPerTimeFrame = ic.options().get<int>(OPTNAME_NOF_ROFS_PER_TF);
    mSeed = ic.options().get<int>(OPTNAME_SEED);

    if (!mSeed) {
      std::random_device rd;
      mSeed = rd();
    }
    mMersenneTwister.seed(mSeed);

    LOGP(INFO,
         "Will generate {:7.2f}% of pads in {:4d} ROFs "
         "per timeframe, for {:4d} timeframes",
         mOccupancy * 100.0, mNofRofPerTimeFrame, mMaxNofTimeFrames);
  }

  void run(ProcessingContext& pc)
  {
    if (mNofProcessedTFs >= mMaxNofTimeFrames) {
      pc.services().get<ControlService>().endOfStream();
      return;
    }

    if (shouldProcess()) {
      incNofProcessedTFs();
      std::vector<Digit> digits;
      std::vector<ROFRecord> rofs;
      int ndigits{0};

      for (auto i = 0; i < mNofRofPerTimeFrame; i++) {
        auto n = generateRandomDigits(mOccupancy, digits);
        o2::InteractionRecord ir{mBC, mOrbit};
        rofs.emplace_back(ir, ndigits, n);
        ndigits += n;
        mOrbit++;
      }
      pc.outputs().snapshot(OutputRef{"rofs"}, rofs);
      pc.outputs().snapshot(OutputRef{"digits"}, digits);
      printSummary(digits, rofs);
      printFull(digits, rofs);
    }
    incTFid();
  }

  /**
 * Populate the digits vector with (Nmch * occupancy) digits
 * where Nmch is the total number of channels in MCH (1064008)
 * 
 * Each member variable of the digit is uniformly distributed within
 * its expected range, as an attempt to maximize the entropy of the
 * generated digits, which can be handy to test the efficiency of the
 * entropy encoder used to create the Compressed Time Frame (CTF) for instance.
 *
 * @param occupancy is a number between 0 and 1.
 * @digits a vector where the generated digits will be appended. 
 * That vector is not cleared by this function, so digits can 
 * be accumulated if need be.
 * @returns the number of digits added to the input digits vector.
 */
  int generateRandomDigits(float occupancy,
                           std::vector<Digit>& digits)
  {
    int n{0};
    std::uniform_int_distribution<int32_t> adc{0, 1024 * 1024};
    std::uniform_int_distribution<int32_t> tfTime{0, 512 * 3564};
    std::uniform_int_distribution<int32_t> nofSamples{0, 1023};
    std::uniform_real_distribution<float> sat{0.0, 1.0};

    auto deids = getDeIds();
    auto nofPadsPerDe = getNofPads();
    auto& mt = mMersenneTwister;

    for (auto i = 0; i < deids.size(); i++) {
      auto deid = deids[i];
      int16_t nofPads = static_cast<int16_t>(nofPadsPerDe[i]);
      std::uniform_int_distribution<int16_t> padid{0, nofPads};
      int nch = nofPads * occupancy;
      for (int i = 0; i < nch; i++) {
        auto p = padid(mt);
        bool isSaturated = (sat(mt) > 0.9);
        digits.emplace_back(deid, p, tfTime(mt), adc(mt), nofSamples(mt), isSaturated);
        ++n;
      }
    }
    return n;
  }

 private:
  float mOccupancy = 1.0;
  uint16_t mBC = 0;
  uint32_t mOrbit = 0;
  int mNofRofPerTimeFrame = 100;
  int mSeed = 0;
  std::mt19937 mMersenneTwister;
};

#include "Framework/runDataProcessing.h"

using namespace o2::mch;

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  auto commonOptions = o2::mch::io::getCommonOptions();
  auto options = Options{
    {OPTNAME_OCCUPANCY, VariantType::Float, 0.01f, {"occupancy (fraction of fired pad per DE per ROF)"}},
    {OPTNAME_SEED, VariantType::Int, 0, {"seed for number generator (if 0 use default_seed)"}},
    {OPTNAME_NOF_ROFS_PER_TF, VariantType::Int, 100, {"number of ROFs per timeframe"}}};
  options.insert(options.end(), commonOptions.begin(), commonOptions.end());

  return WorkflowSpec{
    DataProcessorSpec{
      "mch-digits-random-generator",
      Inputs{},
      Outputs{OutputSpec{{"digits"}, "MCH", "DIGITS", 0, Lifetime::Timeframe},
              OutputSpec{{"rofs"}, "MCH", "DIGITROFS", 0, Lifetime::Timeframe}},
      AlgorithmSpec{adaptFromTask<DigitGenerator>()},
      options}};
}
