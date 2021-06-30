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

namespace o2
{
namespace hmpid
{

// define HMPID hit type
// class Hit : public o2::BasicXYZQHit<float>
class HitType : public o2::BasicXYZEHit<float>
{
 public:
  using BasicXYZEHit<float>::BasicXYZEHit;

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

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDDATAFORMAT_HIT_H_ */
