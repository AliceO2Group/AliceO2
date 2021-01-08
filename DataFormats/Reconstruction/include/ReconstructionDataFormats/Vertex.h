// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_VERTEX_H
#define ALICEO2_VERTEX_H

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonArray.h"
#include <MathUtils/Cartesian.h>

#include "CommonDataFormat/TimeStamp.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <iosfwd>
#endif

namespace o2
{
namespace dataformats
{

// Base primary vertex class, with position, error
class VertexBase
{
 public:
  enum CovElems : int { kCovXX,
                        kCovXY,
                        kCovYY,
                        kCovXZ,
                        kCovYZ,
                        kCovZZ };
  static constexpr int kNCov = 6;
  GPUdDefault() VertexBase() = default;
  GPUdDefault() ~VertexBase() = default;
  GPUd() VertexBase(const math_utils::Point3D<float>& pos, const gpu::gpustd::array<float, kNCov>& cov) : mPos(pos), mCov(cov)
  {
  }

#ifndef GPUCA_GPUCODE_DEVICE
  void print() const;
  std::string asString() const;
#endif

  // getting the cartesian coordinates and errors
  GPUd() float getX() const { return mPos.X(); }
  GPUd() float getY() const { return mPos.Y(); }
  GPUd() float getZ() const { return mPos.Z(); }
  GPUd() float getSigmaX2() const { return mCov[kCovXX]; }
  GPUd() float getSigmaY2() const { return mCov[kCovYY]; }
  GPUd() float getSigmaZ2() const { return mCov[kCovZZ]; }
  GPUd() float getSigmaXY() const { return mCov[kCovXY]; }
  GPUd() float getSigmaXZ() const { return mCov[kCovXZ]; }
  GPUd() float getSigmaYZ() const { return mCov[kCovYZ]; }
  GPUd() const gpu::gpustd::array<float, kNCov>& getCov() const { return mCov; }

  GPUd() math_utils::Point3D<float> getXYZ() const { return mPos; }
  GPUd() math_utils::Point3D<float>& getXYZ() { return mPos; }

  GPUd() void setX(float x) { mPos.SetX(x); }
  GPUd() void setY(float y) { mPos.SetY(y); }
  GPUd() void setZ(float z) { mPos.SetZ(z); }

  GPUd() void setXYZ(float x, float y, float z)
  {
    setX(x);
    setY(y);
    setZ(z);
  }
  GPUd() void setPos(const math_utils::Point3D<float>& p) { mPos = p; }

  GPUd() void setSigmaX2(float v) { mCov[kCovXX] = v; }
  GPUd() void setSigmaY2(float v) { mCov[kCovYY] = v; }
  GPUd() void setSigmaZ2(float v) { mCov[kCovZZ] = v; }
  GPUd() void setSigmaXY(float v) { mCov[kCovXY] = v; }
  GPUd() void setSigmaXZ(float v) { mCov[kCovXZ] = v; }
  GPUd() void setSigmaYZ(float v) { mCov[kCovYZ] = v; }
  GPUd() void setCov(float sxx, float sxy, float syy, float sxz, float syz, float szz)
  {
    setSigmaX2(sxx);
    setSigmaY2(syy);
    setSigmaZ2(szz);
    setSigmaXY(sxy);
    setSigmaXZ(sxz);
    setSigmaYZ(syz);
  }
  GPUd() void setCov(const gpu::gpustd::array<float, kNCov>& cov) { mCov = cov; }

 protected:
  math_utils::Point3D<float> mPos{0., 0., 0.}; ///< cartesian position
  gpu::gpustd::array<float, kNCov> mCov{};     ///< errors, see CovElems enum

  ClassDefNV(VertexBase, 1);
};

// Base primary vertex class, with position, error, N candidates and flags field
// The Stamp template parameter allows to define vertex (time)stamp in different
// formats (ITS ROFrame ID, real time + error etc)

template <typename Stamp>
class Vertex : public VertexBase
{
 public:
  using ushort = unsigned short;
  enum Flags : ushort {
    TimeValidated = 0x1 << 0, // Flag that the vertex was validated by external time measurement (e.g. FIT)
    FlagsMask = 0xffff
  };

  GPUdDefault() Vertex() = default;
  GPUdDefault() ~Vertex() = default;
  GPUd() Vertex(const math_utils::Point3D<float>& pos, const gpu::gpustd::array<float, kNCov>& cov, ushort nCont, float chi2)
    : VertexBase(pos, cov), mNContributors(nCont), mChi2(chi2)
  {
  }

  GPUd() ushort getNContributors() const { return mNContributors; }
  GPUd() void setNContributors(ushort v) { mNContributors = v; }
  GPUd() void addContributor() { mNContributors++; }

  GPUd() ushort getFlags() const { return mBits; }
  GPUd() bool isFlagSet(uint f) const { return mBits & (FlagsMask & f); }
  GPUd() void setFlags(ushort f) { mBits |= FlagsMask & f; }
  GPUd() void resetFrags(ushort f = FlagsMask) { mBits &= ~(FlagsMask & f); }

  GPUd() void setChi2(float v) { mChi2 = v; }
  GPUd() float getChi2() const { return mChi2; }

  GPUd() const Stamp& getTimeStamp() const { return mTimeStamp; }
  GPUd() Stamp& getTimeStamp() { return mTimeStamp; }
  GPUd() void setTimeStamp(const Stamp& v) { mTimeStamp = v; }

 protected:
  float mChi2 = 0;           ///< chi2 or quality of tracks to vertex attachment
  ushort mNContributors = 0; ///< N contributors
  ushort mBits = 0;          ///< bit field for flags
  Stamp mTimeStamp;          ///< vertex time-stamp

  ClassDefNV(Vertex, 3);
};

#ifndef GPUCA_GPUCODE_DEVICE
std::ostream& operator<<(std::ostream& os, const o2::dataformats::VertexBase& v);
#endif

} // namespace dataformats
} // namespace o2
#endif
