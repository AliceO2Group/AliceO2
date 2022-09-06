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

#ifndef ALICEO2_VERTEX_H
#define ALICEO2_VERTEX_H

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonArray.h"
#include <MathUtils/Cartesian.h>

#include "CommonDataFormat/TimeStamp.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <iosfwd>
#include <string>
#include <type_traits>
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
  GPUhdDefault() VertexBase() = default;
  GPUhdDefault() ~VertexBase() = default;
  GPUhd() VertexBase(const math_utils::Point3D<float>& pos, const gpu::gpustd::array<float, kNCov>& cov) : mPos(pos), mCov(cov)
  {
  }

#if !defined(GPUCA_NO_FMT) && !defined(GPUCA_GPUCODE_DEVICE)
  void print() const;
  std::string asString() const;
#endif

  // getting the cartesian coordinates and errors
  GPUhd() float getX() const { return mPos.X(); }
  GPUhd() float getY() const { return mPos.Y(); }
  GPUhd() float getZ() const { return mPos.Z(); }
  GPUd() float getSigmaX2() const { return mCov[kCovXX]; }
  GPUd() float getSigmaY2() const { return mCov[kCovYY]; }
  GPUd() float getSigmaZ2() const { return mCov[kCovZZ]; }
  GPUd() float getSigmaXY() const { return mCov[kCovXY]; }
  GPUd() float getSigmaXZ() const { return mCov[kCovXZ]; }
  GPUd() float getSigmaYZ() const { return mCov[kCovYZ]; }
  GPUd() float getSigmaX() const { return gpu::CAMath::Sqrt(getSigmaX2()); }
  GPUd() float getSigmaY() const { return gpu::CAMath::Sqrt(getSigmaY2()); }
  GPUd() float getSigmaZ() const { return gpu::CAMath::Sqrt(getSigmaZ2()); }

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
  GPUd() void setSigmaX(float val) { setSigmaX2(val * val); }
  GPUd() void setSigmaY(float val) { setSigmaY2(val * val); }
  GPUd() void setSigmaZ(float val) { setSigmaZ2(val * val); }

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

  bool operator==(const VertexBase& other) const;
  bool operator!=(const VertexBase& other) const { return !(*this == other); }

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

  GPUhdDefault() Vertex() = default;
  GPUhdDefault() ~Vertex() = default;
  GPUhd() Vertex(const math_utils::Point3D<float>& pos, const gpu::gpustd::array<float, kNCov>& cov, ushort nCont, float chi2)
    : VertexBase(pos, cov), mChi2(chi2), mNContributors(nCont)
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

#if !defined(GPUCA_GPUCODE_DEVICE) && !defined(GPUCA_NO_FMT)
std::ostream& operator<<(std::ostream& os, const o2::dataformats::VertexBase& v);
#endif

} // namespace dataformats

#ifndef GPUCA_GPUCODE_DEVICE
/// Defining PrimaryVertex explicitly as messageable
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::dataformats::VertexBase> : std::true_type {
};
template <>
struct is_messageable<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>> : std::true_type {
};
template <>
struct is_messageable<o2::dataformats::Vertex<o2::dataformats::TimeStamp<float>>> : std::true_type {
};
template <>
struct is_messageable<o2::dataformats::Vertex<o2::dataformats::TimeStampWithError<float, float>>> : std::true_type {
};
} // namespace framework
#endif

} // namespace o2
#endif
