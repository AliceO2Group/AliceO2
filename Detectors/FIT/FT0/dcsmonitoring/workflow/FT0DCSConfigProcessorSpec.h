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

/// \file FT0DCSConfigProcessorSpec.cxx
/// \brief FT0 processor spec for DCS configurations
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FT0_DCSCONFIGPROCESSOR_H
#define O2_FT0_DCSCONFIGPROCESSOR_H

#include "FITDCSMonitoring/FITDCSConfigProcessorSpec.h"
#include "FT0DCSMonitoring/FT0DCSConfigReader.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/WorkflowSpec.h"
#include "Headers/DataHeader.h"

#include <string>
#include <vector>

namespace o2
{
namespace ft0
{

class FT0DCSConfigProcessor : public o2::fit::FITDCSConfigProcessor
{
  // Example of how to use another DCS config reader (subclass of o2::fit::FITDCSConfigReader)
 public:
  FT0DCSConfigProcessor(const std::string& detectorName, const o2::header::DataDescription& dataDescriptionDChM)
    : o2::fit::FITDCSConfigProcessor(detectorName, dataDescriptionDChM) {}

 protected:
  void initDCSConfigReader() override
  {
    mDCSConfigReader = std::make_unique<FT0DCSConfigReader>(FT0DCSConfigReader());
  }
};

} // namespace ft0

namespace framework
{

DataProcessorSpec getFT0DCSConfigProcessorSpec()
{
  o2::header::DataDescription ddDChM = "FT0_DCHM";
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, ddDChM}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, ddDChM}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "ft0-dcs-config-processor",
    Inputs{{"inputConfig", o2::header::gDataOriginFT0, "DCS_CONFIG_FILE", Lifetime::Timeframe},
           {"inputConfigFileName", o2::header::gDataOriginFT0, "DCS_CONFIG_NAME", Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::ft0::FT0DCSConfigProcessor>("FT0", ddDChM)},
    Options{{"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"filename-dchm", VariantType::String, "FT0-deadchannels.txt", {"Dead channel map file name"}}}};
}

} // namespace framework
} // namespace o2

#endif // O2_FT0_DCSCONFIGPROCESSOR_H
