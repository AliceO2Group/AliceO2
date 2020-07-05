// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/Vertex.h"
#include <iostream>

namespace o2
{
namespace dataformats
{

#ifndef ALIGPU_GPUCODE
std::ostream& operator<<(std::ostream& os, const o2::dataformats::VertexBase& v)
{
  // stream itself
  os << std::scientific << "Vertex X: " << v.getX() << " Y: " << v.getY() << " Z: " << v.getZ()
     << " Cov.mat:\n"
     << v.getSigmaX2() << '\n'
     << v.getSigmaXY() << ' ' << v.getSigmaY2() << '\n'
     << v.getSigmaXZ() << ' ' << v.getSigmaYZ() << ' ' << v.getSigmaZ2() << '\n';
  return os;
}

void VertexBase::print() const
{
  std::cout << *this << std::endl;
}
#endif

} // namespace dataformats
} // namespace o2
