// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/DCA.h"
#include <iostream>

/// \author ruben.shahoyan@cern.ch
/// \brief  class for distance of closest approach to vertex

namespace o2
{
namespace dataformats
{

#ifndef GPUCA_GPUCODE_DEVICE
std::ostream& operator<<(std::ostream& os, const o2::dataformats::DCA& d)
{
  // stream itself
  os << "DCA YZ {" << d.getY() << ", " << d.getZ() << "} Cov {" << d.getSigmaY2() << ", " << d.getSigmaYZ() << ", " << d.getSigmaZ2() << "}";
  return os;
}
#endif

void DCA::print() const
{
#ifndef GPUCA_GPUCODE_DEVICE
  std::cout << *this << '\n';
#endif
}

} // namespace dataformats
} // namespace o2
