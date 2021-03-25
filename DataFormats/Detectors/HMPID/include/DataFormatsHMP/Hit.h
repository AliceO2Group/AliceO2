// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   Digit.h
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 15/02/2021

//    History
//
#ifndef DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_HIT_H_
#define DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_HIT_H_

#include "SimulationDataFormat/BaseHits.h" // for BasicXYZEHit
#include "CommonUtils/ShmAllocator.h"
#include "SimulationDataFormat/Stack.h"
#include "TVector3.h"

namespace o2
{
namespace hmpid
{
namespace raw
{

// define HMPID hit type
 // class Hit : public o2::BasicXYZQHit<float>
class HitType : public o2::BasicXYZEHit<float>
{
  public:
    using BasicXYZEHit<float>::BasicXYZEHit;

  ClassDef(HitType, 1);
};

} // namespace raw
} // namespace hmpid
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::hmpid::raw::HitType> : public o2::utils::ShmAllocator<o2::hmpid::raw::HitType>
{
};
} // namespace std
#endif

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_HIT_H_ */
