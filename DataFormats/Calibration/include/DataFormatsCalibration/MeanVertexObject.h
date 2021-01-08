// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef MEAN_VERTEX_OBJECT_H_
#define MEAN_VERTEX_OBJECT_H_

#include <array>
#include "Rtypes.h"

namespace o2
{
namespace dataformats
{
class MeanVertexObject
{

 public:
  MeanVertexObject(float x, float y, float z, float sigmax, float sigmay, float sigmaz)
  {
    mPos[0] = x;
    mPos[1] = y;
    mPos[2] = z;
    mSigma[0] = sigmax;
    mSigma[1] = sigmay;
    mSigma[2] = sigmaz;
  }
  MeanVertexObject(std::array<float, 3> pos, std::array<float, 3> sigma)
  {
    for (int i = 0; i < 3; i++) {
      mPos[i] = pos[i];
      mSigma[i] = sigma[i];
    }
  }
  MeanVertexObject() = default;
  ~MeanVertexObject() = default;
  MeanVertexObject(const MeanVertexObject& other) = default;
  MeanVertexObject(MeanVertexObject&& other) = default;
  MeanVertexObject& operator=(MeanVertexObject& other) = default;
  MeanVertexObject& operator=(MeanVertexObject&& other) = default;

  void setX(float val) { mPos[0] = val; }
  void setY(float val) { mPos[1] = val; }
  void setZ(float val) { mPos[2] = val; }
  void setPos(std::array<float, 3> val)
  {
    for (int i = 0; i < 3; i++) {
      mPos[i] = val[i];
    }
  }

  float getX() const { return mPos[0]; }
  float getY() const { return mPos[1]; }
  float getZ() const { return mPos[2]; }
  const std::array<float, 3>& getPos() const { return mPos; }

  void setSigmaX(float val) { mSigma[0] = val; }
  void setSigmaY(float val) { mSigma[1] = val; }
  void setSigmaZ(float val) { mSigma[2] = val; }
  void setSigma(std::array<float, 3> val)
  {
    for (int i = 0; i < 3; i++) {
      mSigma[i] = val[i];
    }
  }

  float getSigmaX() const { return mSigma[0]; }
  float getSigmaY() const { return mSigma[1]; }
  float getSigmaZ() const { return mSigma[2]; }
  const std::array<float, 3>& getSigma() const { return mSigma; }

 private:
  std::array<float, 3> mPos;   // position of mean vertex
  std::array<float, 3> mSigma; // sigma of mean vertex

  ClassDefNV(MeanVertexObject, 1);
};
} // namespace dataformats
} // namespace o2

#endif
