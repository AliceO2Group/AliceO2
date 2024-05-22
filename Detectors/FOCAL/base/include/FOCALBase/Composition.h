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
  Composition(const Composition& comp);
  Composition& operator=(const Composition& comp);
  ~Composition();

  void SetCompositionParameters(std::string material, int layer, int stack, int id,
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
  void SetLayerNumber(int layer) { mLayer = layer; }
  void SetId(int id) { mId = id; }
  void SetCenterZ(float val) { mCenterZ = val; }

  std::string Material() const { return mMaterial; }
  int Layer() const { return mLayer; }
  int Stack() const { return mStack; }
  int Id() const { return mId; }
  float CenterX() const { return mCenterX; }
  float CenterY() const { return mCenterY; }
  float CenterZ() const { return mCenterZ; }
  float SizeX() const { return mSizeX; }
  float SizeY() const { return mSizeY; }
  float SizeZ() const { return mSizeZ; }
  float GetThickness(void) const { return mSizeZ; }

 private:
  std::string mMaterial;
  int mLayer;
  int mStack;
  int mId;
  float mCenterX;
  float mCenterY;
  float mCenterZ;
  float mSizeX;
  float mSizeY;
  float mSizeZ;
};
} // end namespace focal
} // end namespace o2
#endif