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

#ifndef O2_FDD_DCSCONFIGREADER_H
#define O2_FDD_DCSCONFIGREADER_H

#include "FITDCSMonitoring/FITFEEConfigurationReader.h"
#include "DataFormatsFDD/FeeConfiguration.h"

namespace o2::fdd
{
class FDDFEEConfigurationReader : public o2::fit::FITFEEConfigurationReader<FDDFEEConfigurationReader>
{
 public:
  FddFeeConfiguration parseFeeConfiguration(gsl::span<const char> configBuf);
  void parseTriggers(const rapidjson::Value& root, const char* triggersNodeName, TriggersConfig& config);
};

} // namespace o2::fdd

#endif // O2_FV0_DCSCONFIGREADER_H