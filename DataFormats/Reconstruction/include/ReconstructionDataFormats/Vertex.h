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

#include "MathUtils/Cartesian3D.h"
#include "CommonDataFormat/TimeStamp.h"
#ifndef __OPENCL__
#include <array>
#endif
#ifndef ALIGPU_GPUCODE
#include <iomanip>
#include <ios>
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
  VertexBase() = default;
  ~VertexBase() = default;
  VertexBase(const Point3D<float>& pos, const std::array<float, kNCov>& cov) : mPos(pos), mCov(cov)
  {
  }

#ifndef ALIGPU_GPUCODE
  void print() const;
  std::string asString() const;
#endif

  // getting the cartesian coordinates and errors
  float getX() const { return mPos.X(); }
  float getY() const { return mPos.Y(); }
  float getZ() const { return mPos.Z(); }
  float getSigmaX2() const { return mCov[kCovXX]; }
  float getSigmaY2() const { return mCov[kCovYY]; }
  float getSigmaZ2() const { return mCov[kCovZZ]; }
  float getSigmaXY() const { return mCov[kCovXY]; }
  float getSigmaXZ() const { return mCov[kCovXZ]; }
  float getSigmaYZ() const { return mCov[kCovYZ]; }
  const std::array<float, kNCov>& getCov() const { return mCov; }

  Point3D<float> getXYZ() const { return mPos; }
  Point3D<float>& getXYZ() { return mPos; }

  void setX(float x) { mPos.SetX(x); }
  void setY(float y) { mPos.SetY(y); }
  void setZ(float z) { mPos.SetZ(z); }

  void setXYZ(float x, float y, float z)
  {
    setX(x);
    setY(y);
    setZ(z);
  }
  void setPos(const Point3D<float>& p) { mPos = p; }

  void setSigmaX2(float v) { mCov[kCovXX] = v; }
  void setSigmaY2(float v) { mCov[kCovYY] = v; }
  void setSigmaZ2(float v) { mCov[kCovZZ] = v; }
  void setSigmaXY(float v) { mCov[kCovXY] = v; }
  void setSigmaXZ(float v) { mCov[kCovXZ] = v; }
  void setSigmaYZ(float v) { mCov[kCovYZ] = v; }
  void setCov(float sxx, float sxy, float syy, float sxz, float syz, float szz)
  {
    setSigmaX2(sxx);
    setSigmaY2(syy);
    setSigmaZ2(szz);
    setSigmaXY(sxy);
    setSigmaXZ(sxz);
    setSigmaYZ(syz);
  }
  void setCov(const std::array<float, kNCov>& cov) { mCov = cov; }

 protected:
  Point3D<float> mPos{0., 0., 0.}; ///< cartesian position
  std::array<float, kNCov> mCov{}; ///< errors, see CovElems enum

  ClassDefNV(VertexBase, 1);
};

// Base primary vertex class, with position, error, N candidates and flags field
// The Stamp template parameter allows to define vertex (time)stamp in different
// formats (ITS ROFrame ID, real time + error etc)

template <typename Stamp = o2::dataformats::TimeStamp<int>>
class Vertex : public VertexBase
{
 public:
  using ushort = unsigned short;
  static ushort constexpr FlagsMask = 0xffff;

  Vertex() = default;
  ~Vertex() = default;
  Vertex(const Point3D<float>& pos, const std::array<float, kNCov>& cov, ushort nCont, float chi2)
    : VertexBase(pos, cov), mNContributors(nCont), mChi2(chi2)
  {
  }

#ifndef ALIGPU_GPUCODE
  void print() const;
#endif

  ushort getNContributors() const { return mNContributors; }
  void setNContributors(ushort v) { mNContributors = v; }
  void addContributor() { mNContributors++; }

  ushort getBits() const { return mBits; }
  bool isBitSet(int bit) const { return mBits & (FlagsMask & (0x1 << bit)); }
  void setBits(ushort b) { mBits = b; }
  void setBit(int bit) { mBits |= FlagsMask & (0x1 << bit); }
  void resetBit(int bit) { mBits &= ~(FlagsMask & (0x1 << bit)); }

  void setChi2(float v) { mChi2 = v; }
  float getChi2() const { return mChi2; }

  const Stamp& getTimeStamp() const { return mTimeStamp; }
  Stamp& getTimeStamp() { return mTimeStamp; }
  void setTimeStamp(const Stamp& v) { mTimeStamp = v; }

 private:
  float mChi2 = 0;               ///< chi2 or quality of tracks to vertex attachment
  ushort mNContributors = 0;     ///< N contributors
  ushort mBits = 0;              ///< bit field for flags
  Stamp mTimeStamp;              ///< vertex time-stamp

  ClassDefNV(Vertex, 2);
};

#ifndef ALIGPU_GPUCODE
std::ostream& operator<<(std::ostream& os, const o2::dataformats::VertexBase& v);

template <typename Stamp>
inline std::ostream& operator<<(std::ostream& os, const o2::dataformats::Vertex<Stamp>& v)
{
  // stream itself
  os << (const VertexBase&)v << " "
     << "NCont: " << v.getNContributors() << " Chi2: " << v.getChi2() << " TimeStamp: " << v.getTimeStamp();
  return os;
}

template <typename Stamp>
void Vertex<Stamp>::print() const
{
  std::cout << *this << std::endl;
}
#endif

} // namespace dataformats
} // namespace o2
#endif
