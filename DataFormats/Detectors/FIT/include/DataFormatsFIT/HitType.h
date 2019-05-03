// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HitType.h
/// \brief Definition of the FIT hits class

#ifndef ALICEO2_FIT_HITTYPE_H_
#define ALICEO2_FIT_HITTYPE_H_

#include "SimulationDataFormat/BaseHits.h"
#include "SimulationDataFormat/Stack.h"
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace fit
{
class HitType : public o2::BasicXYZEHit<float>
{
 public:
  using BasicXYZEHit<float>::BasicXYZEHit;
};

} // namespace fit

} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::fit::HitType> : public o2::utils::ShmAllocator<o2::fit::HitType>
{
};
} // namespace std
#endif
#endif
