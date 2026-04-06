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

/// \file Hit.h
/// \brief Definition of the TRK Hit class

#ifndef ALICEO2_TRK_HIT_H_
#define ALICEO2_TRK_HIT_H_

#include "ITSMFTSimulation/Hit.h"

namespace o2::trk
{
class Hit : public o2::itsmft::Hit
{
 public:
  using o2::itsmft::Hit::Hit; // Inherit constructors
};
} // namespace o2::trk

#endif
