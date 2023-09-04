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

#include "MCHDigitFiltering/DigitFilteringSpec.h"

#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/OutputSpec.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "MCHBase/SanityCheck.h"
#include "MCHStatus/StatusMap.h"
#include "MCHDigitFiltering/DigitFilter.h"
#include "MCHDigitFiltering/DigitFilterParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <fmt/format.h>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace o2::dataformats;
using namespace o2::framework;

namespace o2::mch
{
class DigitFilteringTask
{
 public:
  DigitFilteringTask(bool useMC, bool useStatusMap) : mUseMC{useMC}, mUseStatusMap{useStatusMap} {}

  void init(InitContext& ic)
  {
    mSanityCheck = DigitFilterParam::Instance().sanityCheck;
    mMinADC = DigitFilterParam::Instance().minADC;
    mRejectBackground = DigitFilterParam::Instance().rejectBackground;
    mStatusMask = DigitFilterParam::Instance().statusMask;
    mTimeCalib = DigitFilterParam::Instance().timeOffset;
    auto stop = [this]() {
      LOG(info) << "digit filtering duration = "
                << std::chrono::duration<double, std::milli>(mElapsedTime).count() << " ms";
    };
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);
  }

  void shiftDigitsTime(gsl::span<ROFRecord> rofs, gsl::span<Digit> digits)
  {
    for (auto i = 0; i < rofs.size(); i++) {
      ROFRecord& rof = rofs[i];
      rof.getBCData() += mTimeCalib;
    }

    for (auto i = 0; i < digits.size(); i++) {
      Digit& d = digits[i];
      d.setTime(d.getTime() + mTimeCalib);
    }
  }

  void run(ProcessingContext& pc)
  {
    StatusMap defaultStatusMap;

    // get input
    auto iRofs = pc.inputs().get<gsl::span<ROFRecord>>("rofs");
    auto iDigits = pc.inputs().get<gsl::span<Digit>>("digits");
    auto iLabels = mUseMC ? pc.inputs().get<MCTruthContainer<MCCompLabel>*>("labels") : nullptr;
    auto statusMap = mUseStatusMap && pc.inputs().isValid("statusmap") ? pc.inputs().get<StatusMap*>("statusmap") : nullptr;

    bool abort{false};

    auto tStart = std::chrono::high_resolution_clock::now();

    if (mSanityCheck) {
      LOGP(info, "performing sanity checks");
      auto error = sanityCheck(iRofs, iDigits);

      if (!isOK(error)) {
        if (error.nofOutOfBounds > 0) {
          LOGP(error, asString(error));
          LOGP(error, "in a TF with {} rofs and {} digits", iRofs.size(), iDigits.size());
          abort = true;
        }
      }
    }

    // create the output messages
    auto& oRofs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"rofs"});
    auto& oDigits = pc.outputs().make<std::vector<Digit>>(OutputRef{"digits"});
    auto oLabels = mUseMC ? &pc.outputs().make<MCTruthContainer<MCCompLabel>>(OutputRef{"labels"}) : nullptr;

    if (!abort) {
      bool selectSignal = false;

      mIsGoodDigit = createDigitFilter(mMinADC,
                                       mRejectBackground,
                                       selectSignal,
                                       statusMap ? *statusMap : defaultStatusMap,
                                       mStatusMask);
      // at digit filtering stage it is important to keep the 3rd parameter
      // (selectSignal) to false in the call above : the idea is to not cut
      // too much on the tails of the charge distributions otherwise
      // the clustering resolution will suffer.
      // That's why we only apply the "reject background" filter, which
      // is a loose background cut that does not penalize the signal
      int cursor{0};
      for (const auto& irof : iRofs) {
        const auto digits = iDigits.subspan(irof.getFirstIdx(), irof.getNEntries());

        // filter the digits from the current ROF
        for (auto i = 0; i < digits.size(); i++) {
          const auto& d = digits[i];
          if (mIsGoodDigit(d)) {
            oDigits.emplace_back(d);
            if (iLabels) {
              oLabels->addElements(oLabels->getIndexedSize(), iLabels->getLabels(i + irof.getFirstIdx()));
            }
          }
        }
        int nofGoodDigits = oDigits.size() - cursor;
        if (nofGoodDigits > 0) {
          // we create an ouput ROF only if at least one digit from
          // the input ROF passed the filtering
          oRofs.emplace_back(ROFRecord(irof.getBCData(),
                                       cursor,
                                       nofGoodDigits,
                                       irof.getBCWidth()));
          cursor += nofGoodDigits;
        }
      }
    }

    auto labelMsg = mUseMC ? fmt::format("| {} labels (out of {})", oLabels->getNElements(), iLabels->getNElements()) : "";

    LOGP(info, "Kept after filtering : {} rofs (out of {}) | {} digits (out of {}) {}",
         oRofs.size(), iRofs.size(),
         oDigits.size(), iDigits.size(),
         labelMsg);

    if (mTimeCalib != 0) {
      shiftDigitsTime(oRofs, oDigits);
    }

    if (abort) {
      LOGP(error, "Sanity check failed");
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    mElapsedTime += tEnd - tStart;
  }

 private:
  bool mRejectBackground{false};
  bool mSanityCheck{false};
  bool mUseMC{false};
  bool mUseStatusMap{false};
  int mMinADC{1};
  int32_t mTimeCalib{0};
  uint32_t mStatusMask{0};
  DigitFilter mIsGoodDigit;
  std::chrono::duration<double> mElapsedTime{};
};

framework::DataProcessorSpec
  getDigitFilteringSpec(bool useMC,
                        std::string_view specName,
                        std::string_view inputDigitDataDescription,
                        std::string_view outputDigitDataDescription,
                        std::string_view inputDigitRofDataDescription,
                        std::string_view outputDigitRofDataDescription,
                        std::string_view inputDigitLabelDataDescription,
                        std::string_view outputDigitLabelDataDescription)

{
  std::string input =
    fmt::format("digits:MCH/{}/0;rofs:MCH/{}/0",
                inputDigitDataDescription,
                inputDigitRofDataDescription);
  if (useMC) {
    input += fmt::format(";labels:MCH/{}/0", inputDigitLabelDataDescription);
  }

  bool useStatusMap = DigitFilterParam::Instance().statusMask != 0;

  if (useStatusMap) {
    // input += ";statusmap:MCH/STATUSMAP/0?lifetime=sporadic";
    input += ";statusmap:MCH/STATUSMAP/0";
  }

  std::string output =
    fmt::format("digits:MCH/{}/0;rofs:MCH/{}/0",
                outputDigitDataDescription,
                outputDigitRofDataDescription);
  if (useMC) {
    output += fmt::format(";labels:MCH/{}/0", outputDigitLabelDataDescription);
  }

  std::vector<OutputSpec> outputs;
  auto matchers = select(output.c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  return DataProcessorSpec{
    specName.data(),
    Inputs{select(input.c_str())},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitFilteringTask>(useMC, useStatusMap)},
    Options{}};
}
} // namespace o2::mch
