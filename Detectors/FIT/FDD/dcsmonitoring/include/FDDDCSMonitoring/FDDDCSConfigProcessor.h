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

/// \author Wiktor Pierożak <wiktor.pierozak@cern.ch>, AGH University of Krakow, Poland

#ifndef O2_FDD_DCS_CONFIG_PROCESSOR_H
#define O2_FDD_DCS_CONFIG_PROCESSOR_H

#include "FITDCSMonitoring/FITDCSConfigProcessor.h"
#include "FDDDCSMonitoring/FDDFEEConfigurationReader.h"
#include "FITDCSMonitoring/FITHvConfigurationReader.h"

namespace o2::fdd
{
class FDDDCSConfigProcessor : public o2::fit::FITDCSConfigProcessor
{
 public:
  FDDDCSConfigProcessor(const std::string& detectorName, const o2::header::DataDescription& dataDescriptionDChM, const o2::header::DataDescription& dataDescriptionFeeConfig, const o2::header::DataDescription& dataDescriptionHvConfig)
    : o2::fit::FITDCSConfigProcessor(detectorName, dataDescriptionDChM, dataDescriptionFeeConfig, dataDescriptionHvConfig) {}

  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  FDDFEEConfigurationReader mFeeConfigurationReader;
  o2::fit::FITHvConfigurationReader mHvConfigurationReader;
};
} // namespace o2::fdd
#endif