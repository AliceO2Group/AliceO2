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

#ifndef ALICEO2_FOCAL_COMPOSITION_H_
#define ALICEO2_FOCAL_COMPOSITION_H_

#include <string>

namespace o2
{

namespace focal
{

class Composition
{
 public:
  Composition() = default;
  Composition(std::string material, int layer, int stack, int id,
              float cx, float cy, float cz, float dx, float dy, float dz);
  Composition(Composition* comp);
  Composition(const Composition& comp) = default;
  ~Composition() = default;

  void setCompositionParameters(std::string material, int layer, int stack, int id,
                                float cx, float cy, float cz, float dx, float dy, float dz)
  {
    mMaterial = material;
    mLayer = layer;
    mStack = stack;
    mId = id;
    mCenterX = cx;
    mCenterY = cy;
    mCenterZ = cz;
    mSizeX = dx;
    mSizeY = dy;
    mSizeZ = dz;
  };
  void setLayerNumber(int layer) { mLayer = layer; }
  void setId(int id) { mId = id; }
  void setCenterZ(float val) { mCenterZ = val; }

  std::string material() const { return mMaterial; }
  int layer() const { return mLayer; }
  int stack() const { return mStack; }
  int id() const { return mId; }
  float centerX() const { return mCenterX; }
  float centerY() const { return mCenterY; }
  float centerZ() const { return mCenterZ; }
  float sizeX() const { return mSizeX; }
  float sizeY() const { return mSizeY; }
  float sizeZ() const { return mSizeZ; }
  float getThickness(void) const { return mSizeZ; }

 private:
  std::string mMaterial;
  int mLayer = 0;
  int mStack = 0;
  int mId = 0;
  float mCenterX = 0.0;
  float mCenterY = 0.0;
  float mCenterZ = 0.0;
  float mSizeX = 0.0;
  float mSizeY = 0.0;
  float mSizeZ = 0.0;
};
} // end namespace focal
} // end namespace o2
#endif