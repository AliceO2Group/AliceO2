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

#ifndef O2_FT0_DCSCONFIGPROCESSOR_H
#define O2_FT0_DCSCONFIGPROCESSOR_H

#include "FITDCSMonitoring/FITDCSConfigProcessor.h"
#include "FT0DCSMonitoring/FT0FEEConfigurationReader.h"
#include "FITDCSMonitoring/FITHvConfigurationReader.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/WorkflowSpec.h"
#include "Headers/DataHeader.h"

#include <string>
#include <vector>

namespace o2::ft0
{
class FT0DCSConfigProcessor : public o2::fit::FITDCSConfigProcessor
{
 public:
  FT0DCSConfigProcessor(const std::string& detectorName, const o2::header::DataDescription& dataDescriptionDChM, const o2::header::DataDescription& dataDescriptionFeeConfig, const o2::header::DataDescription& dataDescriptionHvConfig)
    : o2::fit::FITDCSConfigProcessor(detectorName, dataDescriptionDChM, dataDescriptionFeeConfig, dataDescriptionHvConfig) {}

  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  FT0FEEConfigurationReader mFeeConfigurationReader;
  o2::fit::FITHvConfigurationReader mHvConfigurationReader;
};
} // namespace o2::ft0
#endif // O2_FT0_DCSCONFIGPROCESSOR_H