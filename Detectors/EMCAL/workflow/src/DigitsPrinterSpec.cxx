// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <vector>

#include "FairLogger.h"

#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "DataFormatsEMCAL/EMCALBlockHeader.h"
#include "EMCALBase/Digit.h"
#include "EMCALWorkflow/DigitsPrinterSpec.h"

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getEmcalDigitsPrinterSpec()
{
  auto initFunction = [](o2::framework::InitContext& ctx) {
    auto processFunction = [](o2::framework::ProcessingContext& pc) {
      // Get the EMCAL block header and check whether it contains digits
      LOG(DEBUG) << "[EMCALDigitsPrinter - process] called";
      auto dataref = pc.inputs().get("digits");
      auto const* emcheader = o2::framework::DataRefUtils::getHeader<o2::emcal::EMCALBlockHeader*>(dataref);
      if (!emcheader->mHasPayload) {
        LOG(DEBUG) << "[EMCALDigitsPrinter - process] No more digits" << std::endl;
        pc.services().get<o2::framework::ControlService>().readyToQuit(false);
        return;
      }

      auto digits = pc.inputs().get<std::vector<o2::emcal::Digit>>("digits");
      std::cout << "[EMCALDigitsPrinter - process] receiveed " << digits.size() << " digits ..." << std::endl;
      if (digits.size()) {
        for (const auto& d : digits) {
          std::cout << "[EMCALDigitsPrinter - process] Channel: " << d.GetTower() << std::endl;
          std::cout << "[EMCALDigitsPrinter - process] Amplitude: " << d.GetAmplitude() << std::endl;
          //std::cout << d << std::endl;
        }
      }
    };

    return processFunction;
  };

  return o2::framework::DataProcessorSpec{ "EMCALDigitsPrinter",
                                           { { "digits", o2::header::gDataOriginEMC, "DIGITS", 0, o2::framework::Lifetime::Timeframe } },
                                           {},
                                           o2::framework::AlgorithmSpec(initFunction) };
}
