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
#include "Framework/Logger.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "DataFormatsCalibration/MeanVertexBiasParam.h"

namespace o2
{
namespace dataformats
{
class MeanVertexObject : public VertexBase
{
 public:
  MeanVertexObject(float x, float y, float z, float sigmax, float sigmay, float sigmaz, float slopeX, float slopeY)
  {
    if (!gMVBias) {
      checkExternalBias();
    }
    setXYZ(x, y, z);
    setSigma({sigmax, sigmay, sigmaz});
    mSlopeX = slopeX;
    mSlopeY = slopeY;
  }

  MeanVertexObject(std::array<float, 3> pos, std::array<float, 3> sigma, float slopeX, float slopeY)
  {
    if (!gMVBias) {
      checkExternalBias();
    }
    math_utils::Point3D<float> p(pos[0], pos[1], pos[2]);
    setPos(p);
    setSigma(sigma);
    mSlopeX = slopeX;
    mSlopeY = slopeY;
  }

  MeanVertexObject()
  {
    if (!gMVBias) {
      checkExternalBias();
    }
  }

  ~MeanVertexObject() = default;
  MeanVertexObject(const MeanVertexObject& other) = default;
  MeanVertexObject(MeanVertexObject&& other) = default;
  MeanVertexObject& operator=(const MeanVertexObject& other) = default;
  MeanVertexObject& operator=(MeanVertexObject&& other) = default;

  void set(int icoord, float val);
  void setSigma(int icoord, float val);
  void setSigma(std::array<float, 3> val)
  {
    setSigmaX(val[0]);
    setSigmaY(val[1]);
    setSigmaZ(val[2]);
  }
  void setSlopeX(float val) { mSlopeX = val; }
  void setSlopeY(float val) { mSlopeY = val; }

  // getting the cartesian coordinates and errors
  float getX() const { return VertexBase::getX() + gMVBias->xyz[0]; }
  float getY() const
  {
    return VertexBase::getY() + gMVBias->xyz[1];
    ;
  }
  float getZ() const
  {
    return VertexBase::getZ() + gMVBias->xyz[2];
    ;
  }
  float getR() const { return gpu::CAMath::Hypot(getX(), getY()); }

  math_utils::Point3D<float> getXYZ() const { return {getX(), getY(), getZ()}; }
  math_utils::Point3D<float> getPos() const { return getXYZ(); }

  float getSlopeX() const { return mSlopeX + gMVBias->slopeX; }
  float getSlopeY() const { return mSlopeY + gMVBias->slopeY; }

  float getXAtZ(float z) const { return getX() + getSlopeX() * (z - getZ()); }
  float getYAtZ(float z) const { return getY() + getSlopeY() * (z - getZ()); }

  void print() const;
  std::string asString() const;

  /// sample a vertex from the MeanVertex parameters
  math_utils::Point3D<float> sample() const;

  VertexBase getMeanVertex(float z) const
  {
    // set z-dependent x,z, assuming that the cov.matrix is already set
    VertexBase v = *this;
    v.setXYZ(getXAtZ(z), getYAtZ(z), z);
    return v;
  }

  void setMeanXYVertexAtZ(VertexBase& v, float z) const
  {
    v = *this;
    v.setX(getXAtZ(z));
    v.setY(getYAtZ(z));
    v.setZ(z);
  }

  const VertexBase getMeanVertex() const
  {
    return getMeanVertex(getZ());
  }

  static void checkExternalBias();

 private:
  float mSlopeX{0.f}; // slope of x = f(z)
  float mSlopeY{0.f}; // slope of y = f(z)

  static const MeanVertexBiasParam* gMVBias;

  ClassDefNV(MeanVertexObject, 2);
};

std::ostream& operator<<(std::ostream& os, const o2::dataformats::MeanVertexObject& o);

} // namespace dataformats
} // namespace o2

#endif
