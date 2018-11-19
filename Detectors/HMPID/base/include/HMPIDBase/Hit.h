// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_HIT_H_
#define DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_HIT_H_

#include "SimulationDataFormat/BaseHits.h"
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace hmpid
{

// define HMPID hit type
class HitType : public o2::BasicXYZEHit<float>
{
 public:
  using Base = o2::BasicXYZEHit<float>;
  using Base::Base;
  ClassDef(HitType, 1);
};

} // namespace hmpid
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::hmpid::HitType> : public o2::utils::ShmAllocator<o2::hmpid::HitType>
{
};
} // namespace std
#endif

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_HIT_H_ */
