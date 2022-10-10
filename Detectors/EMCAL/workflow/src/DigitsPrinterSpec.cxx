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

#include <iostream>
#include <vector>
#include <type_traits>
#include <gsl/span>

#include <fairlogger/Logger.h>

#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "EMCALWorkflow/DigitsPrinterSpec.h"

using namespace o2::emcal::reco_workflow;

template <class InputType>
void DigitsPrinterSpec<InputType>::init(o2::framework::InitContext& ctx)
{
}

template <class InputType>
void DigitsPrinterSpec<InputType>::run(framework::ProcessingContext& pc)
{
  // Get the EMCAL block header and check whether it contains digits
  LOG(debug) << "[EMCALDigitsPrinter - process] called";
  std::string objectbranch;
  if constexpr (std::is_same<InputType, o2::emcal::Digit>::value) {
    objectbranch = "digits";
  } else if constexpr (std::is_same<InputType, o2::emcal::Cell>::value) {
    objectbranch = "cells";
  } else {
    LOG(error) << "Unsupported input type ... ";
    return;
  }

  auto objects = pc.inputs().get<gsl::span<InputType>>(objectbranch);
  auto triggerrecords = pc.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("triggerrecord");
  std::cout << "[EMCALDigitsPrinter - process] receiveed " << objects.size() << " digits from " << triggerrecords.size() << " triggers ..." << std::endl;
  if (triggerrecords.size()) {
    for (const auto& trg : triggerrecords) {
      if (!trg.getNumberOfObjects()) {
        std::cout << "[EMCALDigitsPrinter - process] Trigger does not contain " << objectbranch << ", skipping ..." << std::endl;
        continue;
      }
      std::cout << "[EMCALDigitsPrinter - process] Trigger has " << trg.getNumberOfObjects() << " " << objectbranch << " ..." << std::endl;
      gsl::span<const InputType> objectsTrigger(objects.data() + trg.getFirstEntry(), trg.getNumberOfObjects());
      for (const auto& d : objectsTrigger) {
        std::cout << "[EMCALDigitsPrinter - process] Channel: " << d.getTower() << std::endl;
        std::cout << "[EMCALDigitsPrinter - process] Energy: " << d.getEnergy() << std::endl;
        //std::cout << d << std::endl;
      }
    }
  }
}

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getEmcalDigitsPrinterSpec(std::string inputtype)
{
  if (inputtype == "digits") {
    return o2::framework::DataProcessorSpec{"EMCALDigitsPrinter",
                                            {{"digits", o2::header::gDataOriginEMC, "DIGITS", 0, o2::framework::Lifetime::Timeframe},
                                             {"triggerrecord", o2::header::gDataOriginEMC, "DIGITSTRGR", 0, o2::framework::Lifetime::Timeframe}},
                                            {},
                                            o2::framework::adaptFromTask<o2::emcal::reco_workflow::DigitsPrinterSpec<o2::emcal::Digit>>()};
  } else if (inputtype == "cells") {
    return o2::framework::DataProcessorSpec{"EMCALDigitsPrinter",
                                            {{"cells", o2::header::gDataOriginEMC, "CELLS", 0, o2::framework::Lifetime::Timeframe},
                                             {"triggerrecord", o2::header::gDataOriginEMC, "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe}},
                                            {},
                                            o2::framework::adaptFromTask<o2::emcal::reco_workflow::DigitsPrinterSpec<o2::emcal::Cell>>()};
  }
  throw std::runtime_error("Input type not supported");
}

//template class o2::emcal::reco_workflow::DigitsPrinterSpec<o2::emcal::Digit>;
//template class o2::emcal::reco_workflow::DigitsPrinterSpec<o2::emcal::Cell>;
