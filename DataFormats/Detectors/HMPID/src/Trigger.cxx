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
/// \file   Trigger.cxx
/// \author Antonio Franco - INFN Bari
/// \brief Base Class to manage HMPID Trigger data
/// \version 1.0
/// \date 6/04/2021

/* ------ HISTORY ---------

*/

#include <iostream>
#include "DataFormatsHMP/Trigger.h"

ClassImp(o2::hmpid::Trigger);

namespace o2
{
namespace hmpid
{

// Digit ASCCI format Dump := (Orbit,BC @ LHCtime ns) [first_digit_idx .. last_digit_idx]
std::ostream& operator<<(std::ostream& os, const o2::hmpid::Trigger& d)
{
  os << "(" << d.mIr.orbit << "," << d.mIr.bc << " @ " << d.mIr.bc2ns() << " ns) [" << d.mDataRange.getFirstEntry() << "," << d.mDataRange.getEntries() << "]";
  return os;
};

} // namespace hmpid
} // namespace o2
