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

#include "DataFormatsFV0/RecPoints.h"
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>

using namespace o2::fv0;

gsl::span<const ChannelDataFloat> RecPoints::getBunchChannelData(const gsl::span<const ChannelDataFloat> tfdata) const
{
  // extract the span of channel data for this bunch from the whole TF data
  return ref.getEntries() ? gsl::span<const ChannelDataFloat>(tfdata).subspan(ref.getFirstEntry(), ref.getEntries()) : gsl::span<const ChannelDataFloat>();
}

void ChannelDataFloat::print() const
{
  printf("  Channel=%d | time=%f | charge=%f | adcId=%d\n", channel, time, charge, adcId);
}
