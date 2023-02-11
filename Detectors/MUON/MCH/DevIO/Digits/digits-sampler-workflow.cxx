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

#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DigitIOBaseTask.h"
#include "DigitSampler.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/Variant.h"
#include "Framework/WorkflowSpec.h"
#include "ProgOptions.h"
#include <algorithm>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using namespace o2::framework;

constexpr const char* OPTNAME_INFILE = "infile";
constexpr const char* OPTNAME_MAX_NOF_ROFS = "max-nof-rofs";
constexpr const char* OPTNAME_REPACK_ROFS = "repack-rofs";
constexpr const char* OPTNAME_RUN2 = "run2";

using namespace o2::mch;

class DigitSamplerTask : public io::DigitIOBaseTask
{
 private:
  std::unique_ptr<io::DigitSampler> mDigitSampler;
  std::ifstream mInput;
  bool mReadIsOk = true;
  size_t mMaxNofROFs;
  size_t mNofProcessedROFs{0};
  size_t mMinNumberOfROFsPerTF{1};
  std::vector<ROFRecord> mROFs;
  std::vector<Digit> mDigits;

 public:
  void init(InitContext& ic)
  {
    io::DigitIOBaseTask::init(ic); // init common options
    auto inputFileName = ic.options().get<std::string>(OPTNAME_INFILE);
    mInput.open(inputFileName);
    mDigitSampler = std::make_unique<io::DigitSampler>(mInput);
    mNofProcessedTFs = 0;
    mMaxNofROFs = ic.options().get<int>(OPTNAME_MAX_NOF_ROFS);
    mMinNumberOfROFsPerTF = ic.options().get<int>(OPTNAME_REPACK_ROFS);
  }

  void outputAndClear(DataAllocator& out)
  {
    printSummary(mDigits, mROFs, "-> to output");
    out.snapshot(OutputRef{"rofs"}, mROFs);
    out.snapshot(OutputRef{"digits"}, mDigits);
    mDigits.clear();
    mROFs.clear();
  }

  bool shouldEnd() const
  {
    bool maxTFreached = mNofProcessedTFs >= mMaxNofTimeFrames;
    bool maxROFreached = mNofProcessedROFs >= mMaxNofROFs;
    return !mReadIsOk || maxTFreached || maxROFreached;
  }

  void run(ProcessingContext& pc)
  {
    if (shouldEnd()) {
      // output remaining data if any
      if (mROFs.size() > 0) {
        --mTFid;
        outputAndClear(pc.outputs());
      }
      pc.services().get<ControlService>().endOfStream();
      return;
    }

    std::vector<ROFRecord> rofs;
    std::vector<Digit> digits;
    mReadIsOk = mDigitSampler->read(digits, rofs);
    if (!mReadIsOk) {
      return;
    }

    if (shouldProcess()) {
      incNofProcessedTFs();
      mNofProcessedROFs += rofs.size();
      // append rofs to mROFs, but shift the indices by the amount of digits
      // we have read so far.
      auto offset = mDigits.size();
      std::transform(rofs.begin(), rofs.end(), std::back_inserter(mROFs),
                     [offset](ROFRecord r) {
                       r.setDataRef(r.getFirstIdx() + offset, r.getNEntries());
                       return r;
                     });
      mDigits.insert(mDigits.end(), digits.begin(), digits.end());
      printSummary(mDigits, mROFs);
      printFull(mDigits, mROFs);
    }

    // output if we've accumulated enough ROFs
    if (mROFs.size() >= mMinNumberOfROFsPerTF) {
      outputAndClear(pc.outputs());
    }

    incTFid();
  }
};

o2::framework::DataProcessorSpec getDigitSamplerSpec(const char* specName, bool run2)
{
  std::string spec = fmt::format("digits:MCH/DIGITS{}/0", run2 ? "R2" : "");
  InputSpec itmp = o2::framework::select(spec.c_str())[0];

  auto commonOptions = o2::mch::io::getCommonOptions();
  auto options = Options{
    {OPTNAME_INFILE, VariantType::String, "", {"input file name"}},
    {OPTNAME_MAX_NOF_ROFS, VariantType::Int, std::numeric_limits<int>::max(), {"max number of ROFs to process"}},
    {OPTNAME_REPACK_ROFS, VariantType::Int, 1, {"number of rofs to repack into a timeframe (aka min number of rofs per timeframe"}}};
  options.insert(options.end(), commonOptions.begin(), commonOptions.end());

  return DataProcessorSpec{
    specName,
    Inputs{},
    Outputs{{DataSpecUtils::asOutputSpec(itmp)},
            OutputSpec{{"rofs"}, "MCH", "DIGITROFS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DigitSamplerTask>()},
    options};
}

/** add workflow options. Note that customization needs to be declared
 * before including Framework/runDataProcessing
 */
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back(OPTNAME_RUN2, VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"input digits use Run2 padIds"});
}

#include "Framework/runDataProcessing.h"

//_________________________________________________________________________________________________
WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  return WorkflowSpec{getDigitSamplerSpec("mch-digits-sampler", cc.options().get<bool>(OPTNAME_RUN2))};
}
