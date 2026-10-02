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

#ifndef O2_FT0_DCSCONFIGREADER_H
#define O2_FT0_DCSCONFIGREADER_H

#include "FITDCSMonitoring/FITFEEConfigurationReader.h"
#include "DataFormatsFT0/FeeConfiguration.h"

namespace o2::ft0
{
class FT0FEEConfigurationReader : public o2::fit::FITFEEConfigurationReader<FT0FEEConfigurationReader>
{
 public:
  Ft0FeeConfiguration parseFeeConfiguration(gsl::span<const char> configBuf);
  void parseTriggers(const rapidjson::Value& root, const char* triggersNodeName, TriggersConfig& config);
};

} // namespace o2::ft0

#endif // O2_FT0_DCSCONFIGREADER_H