// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCBase/FECInfo.h"
#include <iostream>

namespace o2
{
namespace TPC
{
std::ostream& FECInfo::print(std::ostream& out) const
{
  out << "FEC in sector [" << int(mIndex)
      //<<"], FEC connector [" << int(mConnector) << "], FEC channel [" << int(mChannel)
      << "], SAMPA chip [" << int(mSampaChip) << "], SAMPA channel [" << int(mSampaChannel) << "]";
  return out;
}

std::ostream& operator<<(std::ostream& out, const TPC::FECInfo& fec)
{
  fec.print(out);
  return out;
}
} // namespace TPC
} // namespace o2
