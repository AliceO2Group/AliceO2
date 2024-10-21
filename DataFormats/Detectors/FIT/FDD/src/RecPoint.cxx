// Copyright 2019-2024 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   RecPoint.cxx
/// \brief  Implementation of the FDD RecPoint class
/// \author Andreas Molander andreas.molander@cern.ch

#include "DataFormatsFDD/RecPoint.h"
#include "Framework/Logger.h"

using namespace o2::fdd;

void ChannelDataFloat::print() const
{
  LOG(info) << "ChannelDataFloat data:";
  LOG(info) << "Channel ID: " << mPMNumber << ", Time (ps): " << mTime << ", Charge (ADC): " << mChargeADC << ", QTC chain: " << adcId;
}

void RecPoint::print() const
{
  LOG(info) << "RecPoint data:";
  LOG(info) << "Collision times: A: " << getCollisionTimeA() << ", C: " << getCollisionTimeC();
  LOG(info) << "Ref first: " << mRef.getFirstEntry() << ", Ref entries: " << mRef.getEntries();
  LOG(info) << "Triggers: " << mTriggers.print();
}
