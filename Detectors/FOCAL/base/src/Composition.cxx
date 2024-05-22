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

#include "FOCALBase/Geometry.h"

using namespace o2::focal;

Composition::Composition(std::string material, int layer, int stack, int id,
                         float cx, float cy, float cz, float dx, float dy, float dz) : mMaterial(material),
                                                                                       mLayer(layer),
                                                                                       mStack(stack),
                                                                                       mId(id),
                                                                                       mCenterX(cx),
                                                                                       mCenterY(cy),
                                                                                       mCenterZ(cz),
                                                                                       mSizeX(dx),
                                                                                       mSizeY(dy),
                                                                                       mSizeZ(dz)
{
  // Default constructor
}

Composition::Composition(Composition* comp) : mMaterial(0),
                                              mLayer(0),
                                              mStack(0),
                                              mId(0),
                                              mCenterX(0),
                                              mCenterY(0),
                                              mCenterZ(0),
                                              mSizeX(0),
                                              mSizeY(0),
                                              mSizeZ(0)
{
  *this = comp;
}

Composition::Composition(const Composition& comp) : mMaterial(comp.mMaterial),
                                                    mLayer(comp.mLayer),
                                                    mStack(comp.mStack),
                                                    mId(comp.mId),
                                                    mCenterX(comp.mCenterX),
                                                    mCenterY(comp.mCenterY),
                                                    mCenterZ(comp.mCenterZ),
                                                    mSizeX(comp.mSizeX),
                                                    mSizeY(comp.mSizeY),
                                                    mSizeZ(comp.mSizeZ)
{
}

Composition& Composition::operator=(const Composition& comp)
{
  if (this != &comp) {
    mMaterial = comp.mMaterial;
    mLayer = comp.mLayer;
    mStack = comp.mStack;
    mId = comp.mId;
    mCenterX = comp.mCenterX;
    mCenterY = comp.mCenterY;
    mCenterZ = comp.mCenterZ;
    mSizeX = comp.mSizeX;
    mSizeY = comp.mSizeY;
    mSizeZ = comp.mSizeZ;
  }
  return *this;
}

Composition::~Composition()
{
  // Default destructor
}