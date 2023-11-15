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

/// \file HitType.h
/// \brief Definition of the FIT hits class

#ifndef ALICEO2_FIT_HITTYPE_H_
#define ALICEO2_FIT_HITTYPE_H_

#include "SimulationDataFormat/BaseHits.h"
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace ft0
{
class HitType : public o2::BasicXYZEHit<float, float, float>
{
 public:
  using BasicXYZEHit<float, float, float>::BasicXYZEHit;
  ClassDefNV(HitType, 1);
};

} // namespace ft0

} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::ft0::HitType> : public o2::utils::ShmAllocator<o2::ft0::HitType>
{
};
} // namespace std
#endif
#endif
