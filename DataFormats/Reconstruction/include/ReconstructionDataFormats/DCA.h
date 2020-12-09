// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DCA_H
#define ALICEO2_DCA_H

#include "GPUCommonRtypes.h"

#ifndef __OPENCL__
#include <array>
#endif
#ifndef GPUCA_ALIGPUCODE
#include <iosfwd>
#endif

/// \author ruben.shahoyan@cern.ch
/// \brief  class for distance of closest approach to vertex

namespace o2
{
namespace dataformats
{

class DCA
{

 public:
  DCA() = default;

  DCA(float y, float z, float syy = 0.f, float syz = 0.f, float szz = 0.f)
  {
    set(y, z, syy, syz, szz);
  }

  void set(float y, float z, float syy, float syz, float szz)
  {
    mY = y;
    mZ = z;
    mCov[0] = syy;
    mCov[1] = syz;
    mCov[2] = szz;
  }

  void set(float y, float z)
  {
    mY = y;
    mZ = z;
  }

  auto getY() const { return mY; }
  auto getZ() const { return mZ; }
  auto getSigmaY2() const { return mCov[0]; }
  auto getSigmaYZ() const { return mCov[1]; }
  auto getSigmaZ2() const { return mCov[2]; }
  const auto& getCovariance() const { return mCov; }

  void print() const;

 private:
  float mY = 0.f;
  float mZ = 0.f;
  std::array<float, 3> mCov; ///< s2y, syz, s2z

  ClassDefNV(DCA, 1);
};

std::ostream& operator<<(std::ostream& os, const DCA& d);

} // namespace dataformats
} // namespace o2

#endif //ALICEO2_DCA_H
