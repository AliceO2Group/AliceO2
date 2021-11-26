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

#include "DigitFilteringSpec.h"

#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/OutputSpec.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "SanityCheck.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <fmt/format.h>
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
  DigitFilteringTask(bool useMC) : mUseMC{useMC} {}

  void init(InitContext& ic)
  {
    mSanityCheck = ic.options().get<bool>("sanity-check");
    mMinADC = ic.options().get<int>("min-adc-value");
  }

  bool isGoodDigit(const Digit& digit) const
  {
    return digit.getADC() > mMinADC;
  }

  void run(ProcessingContext& pc)
  {
    // get input
    auto iRofs = pc.inputs().get<gsl::span<ROFRecord>>("rofs");
    auto iDigits = pc.inputs().get<gsl::span<Digit>>("digits");
    auto iLabels = mUseMC ? pc.inputs().get<MCTruthContainer<MCCompLabel>*>("labels") : nullptr;

    bool abort{false};

    if (mSanityCheck) {
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
      int cursor{0};
      for (auto i = 0; i < iRofs.size(); i++) {
        const ROFRecord& irof = iRofs[i];

        const auto digits = iDigits.subspan(irof.getFirstIdx(), irof.getNEntries());

        // filter the digits from the current ROF
        for (auto i = 0; i < digits.size(); i++) {
          const auto& d = digits[i];
          if (isGoodDigit(d)) {
            oDigits.emplace_back(d);
            if (iLabels) {
              oLabels->addElements(oLabels->getIndexedSize(), iLabels->getLabels(i));
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

    LOGP(info, "Kept after filtering : {} rofs (out of {}) | {} digits (out of {}) {}\n",
         oRofs.size(), iRofs.size(),
         oDigits.size(), iDigits.size(),
         labelMsg);

    if (abort) {
      LOGP(error, "Sanity check failed");
    }
  }

 private:
  bool mSanityCheck;
  bool mUseMC;
  int mMinADC;
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
    AlgorithmSpec{adaptFromTask<DigitFilteringTask>(useMC)},
    Options{
      {"sanity-check", VariantType::Bool, false, {"perform a few sanity checks on input digits"}},
      {"min-adc-value", VariantType::Int, 1, {"minumum ADC value to consider"}}}};
}
} // namespace o2::mch
