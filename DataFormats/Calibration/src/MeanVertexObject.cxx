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

#include "DataFormatsCalibration/MeanVertexObject.h"
#include "TRandom.h"

namespace o2
{
namespace dataformats
{

void MeanVertexObject::set(int icoord, float val)
{
  if (icoord == 0) {
    setX(val);
  } else if (icoord == 1) {
    setY(val);
  } else if (icoord == 2) {
    setZ(val);
  } else {
    LOG(fatal) << "Coordinate out of bound to set vtx " << icoord << ", should be in [0, 2]";
  }
}

void MeanVertexObject::setSigma(int icoord, float val)
{
  if (icoord == 0) {
    setSigmaX2(val);
  } else if (icoord == 1) {
    setSigmaY2(val);
  } else if (icoord == 2) {
    setSigmaZ2(val);
  } else {
    LOG(fatal) << "Coordinate out of bound to set sigma via MeanVtx " << icoord << ", should be in [0, 2]";
  }
}

std::string MeanVertexObject::asString() const
{
  return VertexBase::asString() + fmt::format(" Slopes {{{:+.4e},{:+.4e}}}", mSlopeX, mSlopeY);
}

std::ostream& operator<<(std::ostream& os, const o2::dataformats::MeanVertexObject& o)
{
  // stream itself
  os << o.asString();
  return os;
}

void MeanVertexObject::print() const
{
  std::cout << *this << std::endl;
}

math_utils::Point3D<float> MeanVertexObject::sample() const
{
  // this assumes gaussian sampling
  // first determine z; then x and y
  const auto z = gRandom->Gaus(getZ(), getSigmaZ());
  const auto x = gRandom->Gaus(getXAtZ(z), getSigmaX());
  const auto y = gRandom->Gaus(getYAtZ(z), getSigmaY());
  return math_utils::Point3D<float>(x, y, z);
}

} // namespace dataformats
} // namespace o2
