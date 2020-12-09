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

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonArray.h"

#ifndef GPUCA_GPUCODE_DEVICE
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
  GPUdDefault() DCA() = default;

  GPUd() DCA(float y, float z, float syy = 0.f, float syz = 0.f, float szz = 0.f)
  {
    set(y, z, syy, syz, szz);
  }

  GPUd() void set(float y, float z, float syy, float syz, float szz)
  {
    mY = y;
    mZ = z;
    mCov[0] = syy;
    mCov[1] = syz;
    mCov[2] = szz;
  }

  GPUd() void set(float y, float z)
  {
    mY = y;
    mZ = z;
  }

  GPUd() auto getY() const { return mY; }
  GPUd() auto getZ() const { return mZ; }
  GPUd() auto getSigmaY2() const { return mCov[0]; }
  GPUd() auto getSigmaYZ() const { return mCov[1]; }
  GPUd() auto getSigmaZ2() const { return mCov[2]; }
  GPUd() const auto& getCovariance() const { return mCov; }

  void print() const;

 private:
  float mY = 0.f;
  float mZ = 0.f;
  gpu::gpustd::array<float, 3> mCov; ///< s2y, syz, s2z

  ClassDefNV(DCA, 1);
};

#ifndef GPUCA_GPUCODE_DEVICE
std::ostream& operator<<(std::ostream& os, const DCA& d);
#endif

} // namespace dataformats
} // namespace o2

#endif //ALICEO2_DCA_H
