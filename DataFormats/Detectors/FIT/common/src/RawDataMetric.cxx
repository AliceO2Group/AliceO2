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

#include "DataFormatsFIT/RawDataMetric.h"
#include <Framework/Logger.h>
using namespace o2::fit;
const std::map<unsigned int, std::string> RawDataMetric::sMapBitsToNames = {
  {RawDataMetric::EStatusBits::kIncompletePayload, "IncompletePayload"},
  {RawDataMetric::EStatusBits::kWrongDescriptor, "WrongDescriptor"},
  {RawDataMetric::EStatusBits::kWrongChannelMapping, "WrongChannelMapping"},
  {RawDataMetric::EStatusBits::kEmptyDataBlock, "EmptyDataBlock"},
  {RawDataMetric::EStatusBits::kDecodedDataBlock, "DecodedDataBlock"}};
void RawDataMetric::print() const
{
  LOG(info) << "==============================================================";
  LOG(info) << "Raw data metric: linkID " << static_cast<int>(mLinkID) << " mEPID " << static_cast<int>(mEPID) << " FEEID " << static_cast<int>(mFEEID);
  LOG(info) << "Is registered FEE: " << mIsRegisteredFEE;
  for (const auto& entry : sMapBitsToNames) {
    LOG(info) << entry.second << ": " << mBitStats[entry.first];
  }
  LOG(info) << "==============================================================";
}
RawDataMetric::Status_t RawDataMetric::getAllBitsActivated()
{
  Status_t metricStatus{};
  for (const auto& entry : sMapBitsToNames) {
    metricStatus |= (1 << entry.first);
  }
  return metricStatus;
}
