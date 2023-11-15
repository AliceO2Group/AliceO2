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
#include "DigitSink.h"
#include <fmt/format.h>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InitContext.h"
#include "Framework/Logger.h"
#include "Framework/Variant.h"
#include "ProgOptions.h"
#include <iostream>
#include <sstream>
#include <string>

using namespace o2::framework;

namespace o2::mch::io
{
void DigitIOBaseTask::init(InitContext& ic)
{
  mMaxNofTimeFrames = ic.options().get<int>(OPTNAME_MAX_NOF_TFS);
  mFirstTF = ic.options().get<int>(OPTNAME_FIRST_TF);
  mNofProcessedTFs = 0;
  mPrintDigits = ic.options().get<bool>(OPTNAME_PRINT_DIGITS);
  mPrintTFs = ic.options().get<bool>(OPTNAME_PRINT_TFS);
  if (mPrintDigits || mPrintTFs) {
    fair::Logger::SetConsoleColor(true);
  }
}

void DigitIOBaseTask::printFull(gsl::span<const Digit> digits,
                                gsl::span<const ROFRecord> rofs) const
{
  if (mPrintDigits) {
    std::stringstream str;
    o2::mch::io::DigitSink dw(str);
    dw.write(digits, rofs);
    for (std::string line; std::getline(str, line);) {
      LOG(info) << line;
    }
  }
}

void DigitIOBaseTask::printSummary(gsl::span<const Digit> digits,
                                   gsl::span<const ROFRecord> rofs,
                                   const char* suffix) const
{
  if (mPrintTFs) {
    LOGP(info, "TF {:5d} {:4d} rofs - {:6d} digits - {}", mTFid, rofs.size(), digits.size(), suffix);
  }
}

bool DigitIOBaseTask::shouldProcess() const
{
  return (mTFid >= mFirstTF && mNofProcessedTFs < mMaxNofTimeFrames);
}

void DigitIOBaseTask::incNofProcessedTFs()
{
  ++mNofProcessedTFs;
}

void DigitIOBaseTask::incTFid()
{
  ++mTFid;
}

std::vector<ConfigParamSpec> getCommonOptions()
{
  return {
    {OPTNAME_MAX_NOF_TFS, VariantType::Int, std::numeric_limits<int>::max(), {OPTHELP_MAX_NOF_TFS}},
    {OPTNAME_FIRST_TF, VariantType::Int, 0, {OPTHELP_FIRST_TF}},
    {OPTNAME_PRINT_DIGITS, VariantType::Bool, false, {OPTHELP_PRINT_DIGITS}},
    {OPTNAME_PRINT_TFS, VariantType::Bool, false, {OPTHELP_PRINT_TFS}}};
}

} // namespace o2::mch::io
