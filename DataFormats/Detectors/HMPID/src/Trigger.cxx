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
/// \file   Trigger.cxx
/// \author Antonio Franco - INFN Bari
/// \brief Base Class to manage HMPID Trigger data
/// \version 1.0
/// \date 6/04/2021

/* ------ HISTORY ---------

*/

#include <iostream>
#include "DataFormatsHMP/Trigger.h"

ClassImp(o2::hmpid::Event);

namespace o2
{
namespace hmpid
{

// Digit ASCCI format Dump := (Orbit,BC @ LHCtime ns) [first_digit_idx .. last_digit_idx]
std::ostream& operator<<(std::ostream& os, const o2::hmpid::Event& d)
{
  os << "(" << d.mIr.orbit << "," << d.mIr.bc << " @ " << d.mIr.bc2ns() << " ns) [" << d.mFirstDigit << " .. " << d.mLastDigit << "]";
  return os;
};

} // namespace hmpid
} // namespace o2
