// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonTransform3D.h
/// \author David Rohr

#ifndef GPUCOMMONTRANSFORM3D_H
#define GPUCOMMONTRANSFORM3D_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class Transform3D
{
 public:
  Transform3D() = default;
  Transform3D(float* v)
  {
    for (int i = 0; i < 12; i++) {
      m[i] = v[i];
    }
  }

  GPUd() void Apply(const float* in, float* out) const
  {
    out[0] = m[kXX] * in[0] + m[kXY] * in[1] + m[kXZ] * in[2] + m[kDX];
    out[1] = m[kYX] * in[0] + m[kYY] * in[1] + m[kYZ] * in[2] + m[kDY];
    out[2] = m[kZX] * in[0] + m[kZY] * in[1] + m[kZZ] * in[2] + m[kDZ];
  }

  GPUd() void ApplyVector(const float* in, float* out) const
  {
    out[0] = m[kXX] * in[0] + m[kXY] * in[1] + m[kXZ] * in[2];
    out[1] = m[kYX] * in[0] + m[kYY] * in[1] + m[kYZ] * in[2];
    out[2] = m[kZX] * in[0] + m[kZY] * in[1] + m[kZZ] * in[2];
  }

  GPUd() void ApplyInverse(const float* in, float* out) const
  {
    const float tmp[3] = {in[0] - m[kDX], in[1] - m[kDY], in[2] - m[kDZ]};
    out[0] = m[kXX] * tmp[0] + m[kYX] * tmp[1] + m[kZX] * tmp[2];
    out[1] = m[kXY] * tmp[0] + m[kYY] * tmp[1] + m[kZY] * tmp[2];
    out[2] = m[kXZ] * tmp[0] + m[kYZ] * tmp[1] + m[kZZ] * tmp[2];
  }

  GPUd() void ApplyInverseVector(const float* in, float* out) const
  {
    out[0] = m[kXX] * in[0] + m[kYX] * in[1] + m[kZX] * in[2];
    out[1] = m[kXY] * in[0] + m[kYY] * in[1] + m[kZY] * in[2];
    out[2] = m[kXZ] * in[0] + m[kYZ] * in[1] + m[kZZ] * in[2];
  }

  GPUd() void LocalToMaster(const float* in, float* out) const { ApplyInverse(in, out); }

 private:
  float m[12];

  enum Transform3DMatrixIndex { kXX = 0,
                                kXY = 1,
                                kXZ = 2,
                                kDX = 3,
                                kYX = 4,
                                kYY = 5,
                                kYZ = 6,
                                kDY = 7,
                                kZX = 8,
                                kZY = 9,
                                kZZ = 10,
                                kDZ = 11 };
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
