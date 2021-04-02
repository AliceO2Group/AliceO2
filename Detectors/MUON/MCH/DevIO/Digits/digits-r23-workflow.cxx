// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** Convert Digit's padID from Run2 to Run3.
*/

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHMappingInterface/Segmentation.h"
#include <fmt/format.h>
#include "Framework/Logger.h"

using namespace o2::framework;

#include "Framework/runDataProcessing.h"

struct DigitConverter {
  bool mVerbose{false};

  void init(InitContext& ic)
  {
    mVerbose = ic.options().get<bool>("verbose");
    fair::Logger::SetConsoleColor(true);
  }

  void run(ProcessingContext& pc)
  {
    auto digitsR2 = pc.inputs().get<gsl::span<o2::mch::Digit>>("digits");
    auto& digitsR3 = pc.outputs().make<std::vector<o2::mch::Digit>>(OutputRef{"digits"});
    for (const auto d2 : digitsR2) {
      auto digit = d2;
      int deID = digit.getDetID();
      int digitID = digit.getPadID();
      int manuID = (digitID & 0xFFF000) >> 12;
      int manuCh = (digitID & 0x3F000000) >> 24;

      int padID = o2::mch::mapping::segmentation(deID).findPadByFEE(manuID, manuCh);
      if (mVerbose) {
        LOGP(warn, "DEID {:4d} DIGITID {:10d} MANUID {:4d} CH {:2} PADID {:10d}",
             deID, digitID, manuID, manuCh, padID);
      }
      if (padID < 0) {
        throw std::runtime_error(fmt::format("digitID {} does not exist in the mapping", digitID));
      }
      digit.setPadID(padID);
      digitsR3.push_back(digit);
    }
  }
};

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  WorkflowSpec specs;

  std::string inputConfig = fmt::format("digits:MCH/DIGITSR2/0");
  inputConfig += ";rofs:MCH/DIGITROFS/0";

  DataProcessorSpec padIdConverter{
    "mch-digits-r23",
    Inputs{o2::framework::select(inputConfig.c_str())},
    Outputs{OutputSpec{{"digits"}, "MCH", "DIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DigitConverter>()},
    Options{
      {"verbose", VariantType::Bool, false, {"print ids being converted"}}}};
  specs.push_back(padIdConverter);
  return specs;
}
