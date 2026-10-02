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

/// \file FT0DCSConfigReader.h
/// \brief DCS configuration reader for FT0
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FV0_DCSCONFIGREADER_H
#define O2_FV0_DCSCONFIGREADER_H

#include "FITDCSMonitoring/FITFEEConfigurationReader.h"
#include "DataFormatsFV0/FeeConfiguration.h"

namespace o2::fv0
{
class FV0FEEConfigurationReader : public o2::fit::FITFEEConfigurationReader<FV0FEEConfigurationReader>
{
 public:
  Fv0FeeConfiguration parseFeeConfiguration(gsl::span<const char> configBuf);
  void parseTriggers(const rapidjson::Value& root, const char* triggersNodeName, TriggersConfig& config);
};

} // namespace o2::fv0

#endif // O2_FV0_DCSCONFIGREADER_H