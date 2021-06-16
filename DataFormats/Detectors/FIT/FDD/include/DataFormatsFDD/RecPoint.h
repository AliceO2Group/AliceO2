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

/// \file RecPoint.h
/// \brief Definition of the FDD RecPoint class
#ifndef ALICEO2_FDD_RECPOINT_H
#define ALICEO2_FDD_RECPOINT_H

#include "CommonDataFormat/InteractionRecord.h"

namespace o2
{
namespace fdd
{

struct RecPoint {

  double mMeanTimeFDA = o2::InteractionRecord::DummyTime;
  double mMeanTimeFDC = o2::InteractionRecord::DummyTime;

  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc) from digits

  RecPoint() = default;
  RecPoint(double timeA, double timeC, o2::InteractionRecord iRec) : mMeanTimeFDA(timeA), mMeanTimeFDC(timeC), mIntRecord(iRec) {}

  ClassDefNV(RecPoint, 2);
};
} // namespace fdd
} // namespace o2
#endif
