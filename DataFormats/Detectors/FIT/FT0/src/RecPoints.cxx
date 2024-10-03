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

#include "DataFormatsFT0/RecPoints.h"
#include "FT0Base/Geometry.h"
#include <DataFormatsFT0/ChannelData.h>
#include <DataFormatsFT0/Digit.h>
#include <cmath>
#include <cassert>
#include <iostream>
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>

using namespace o2::ft0;

void ChannelDataFloat::print() const
{

  printf("  ChID% d | CFDtime=%f | QTCampl=%f QTC chain %d\n", ChId, CFDTime, QTCAmpl, ChainQTC);
}

gsl::span<const ChannelDataFloat> RecPoints::getBunchChannelData(const gsl::span<const ChannelDataFloat> tfdata) const
{
  // extract the span of channel data for this bunch from the whole TF data
  return ref.getEntries() ? gsl::span<const ChannelDataFloat>(tfdata).subspan(ref.getFirstEntry(), ref.getEntries()) : gsl::span<const ChannelDataFloat>();
}

void RecPoints::print() const
{
  LOG(info) << "RecPoint data:";
  LOG(info) << "Collision times: mean: " << getCollisionTimeMean() << ", A: " << getCollisionTimeA() << ", C: " << getCollisionTimeC();
  LOG(info) << "Vertex: " << getVertex();
  LOG(info) << "Ref first: " << ref.getFirstEntry() << ", Ref entries: " << ref.getEntries();
  LOG(info) << "Triggers: " << mTriggers.print();
}
