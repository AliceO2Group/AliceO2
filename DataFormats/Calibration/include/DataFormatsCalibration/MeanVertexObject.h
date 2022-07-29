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

#ifndef MEAN_VERTEX_OBJECT_H_
#define MEAN_VERTEX_OBJECT_H_

#include <array>
#include "Rtypes.h"

#include "Framework/Logger.h"
#include "ReconstructionDataFormats/Vertex.h"

namespace o2
{
namespace dataformats
{
class MeanVertexObject : public VertexBase
{

 public:
  MeanVertexObject(float x, float y, float z, float sigmax, float sigmay, float sigmaz, float slopeX, float slopeY)
  {
    math_utils::Point3D<float> p(x, y, z);
    gpu::gpustd::array<float, kNCov> cov;
    cov[CovElems::kCovXX] = sigmax;
    cov[CovElems::kCovYY] = sigmay;
    cov[CovElems::kCovZZ] = sigmaz;
    VertexBase(p, cov);
    mSlopeX = slopeX;
    mSlopeX = slopeY;
  }
  MeanVertexObject(std::array<float, 3> pos, std::array<float, 3> sigma, float slopeX, float slopeY)
  {
    math_utils::Point3D<float> p(pos[0], pos[1], pos[2]);
    gpu::gpustd::array<float, kNCov> cov;
    cov[CovElems::kCovXX] = sigma[0];
    cov[CovElems::kCovYY] = sigma[1];
    cov[CovElems::kCovZZ] = sigma[2];
    VertexBase(p, cov);
    mSlopeX = slopeX;
    mSlopeX = slopeY;
  }
  MeanVertexObject() = default;
  ~MeanVertexObject() = default;
  MeanVertexObject(const MeanVertexObject& other) = default;
  MeanVertexObject(MeanVertexObject&& other) = default;
  MeanVertexObject& operator=(MeanVertexObject& other) = default;
  MeanVertexObject& operator=(MeanVertexObject&& other) = default;

  void set(int icoord, float val)
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
  math_utils::Point3D<float>& getPos() { return getXYZ(); }
  math_utils::Point3D<float> getPos() const { return getXYZ(); }

  void setSigma(int icoord, float val)
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
  void setSigmaX(float val) { setSigmaX2(val); }
  void setSigmaY(float val) { setSigmaY2(val); }
  void setSigmaZ(float val) { setSigmaZ2(val); }
  void setSigma(std::array<float, 3> val)
  {
    setSigmaX2(val[0]);
    setSigmaY2(val[1]);
    setSigmaZ2(val[2]);
  }

  float getSigmaX() const { return getSigmaX2(); }
  float getSigmaY() const { return getSigmaY2(); }
  float getSigmaZ() const { return getSigmaZ2(); }
  const gpu::gpustd::array<float, kNCov>& getSigma() const { return getCov(); }

  void setSlopeX(float val) { mSlopeX = val; }
  void setSlopeY(float val) { mSlopeY = val; }

  float getSlopeX() const { return mSlopeX; }
  float getSlopeY() const { return mSlopeY; }

  float getXAtZ(float z) { return getX() + mSlopeX * (z - getZ()); }
  float getYAtZ(float z) { return getY() + mSlopeY * (z - getZ()); }

  void print() const;
  std::string asString() const;

  VertexBase getMeanVertex(float z)
  {
    VertexBase v = *this;
    v.setXYZ(getXAtZ(z), getYAtZ(z), z);
  }

  const VertexBase& getMeanVertex() const
  {
    return (const VertexBase&)(*this);
  }

 private:
  float mSlopeX{0.f}; // slope of x = f(z)
  float mSlopeY{0.f}; // slope of y = f(z)

  ClassDefNV(MeanVertexObject, 1);
};

std::ostream& operator<<(std::ostream& os, const o2::dataformats::MeanVertexObject& o);

} // namespace dataformats
} // namespace o2

#endif
