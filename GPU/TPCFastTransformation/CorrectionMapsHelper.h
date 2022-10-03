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

/// \file CorrectionMapsHelper.h
/// \brief Helper class to access correction maps
/// \author ruben.shahoian@cern.ch

#ifndef TPC_CORRECTION_MAPS_HELPER_H_
#define TPC_CORRECTION_MAPS_HELPER_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <memory>
#include <vector>
#endif
#include "TPCFastTransform.h"

namespace o2::gpu
{

class CorrectionMapsHelper
{
 public:
  CorrectionMapsHelper() = default;
  ~CorrectionMapsHelper() { clear(); }
  CorrectionMapsHelper(const CorrectionMapsHelper&) = delete;
  void clear();

  GPUd() void Transform(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime = 0) const
  {
    mCorrMap->Transform(slice, row, pad, time, x, y, z, vertexTime, mCorrMapRef, mLumiScale);
  }

  GPUd() void InverseTransformYZtoX(int slice, int row, float y, float z, float& x) const
  {
    mCorrMap->InverseTransformYZtoX(slice, row, y, z, x, mCorrMapRef, mLumiScale);
  }

  GPUd() void InverseTransformYZtoNominalYZ(int slice, int row, float y, float z, float& ny, float& nz) const
  {
  }

  GPUd() const o2::gpu::TPCFastTransform* getCorrMap() const { return mCorrMap; }
  GPUd() const o2::gpu::TPCFastTransform* getCorrMapRef() const { return mCorrMapRef; }

  bool getOwner() const { return mOwner; }

  void setCorrMap(o2::gpu::TPCFastTransform* m);
  void setCorrMapRef(o2::gpu::TPCFastTransform* m);

  void setInstLumi(float v)
  {
    mInstLumi = v;
    mLumiScale = mMeanLumi ? mInstLumi / mMeanLumi : 0.f;
  }
  void setMeanLumi(float v)
  {
    mMeanLumi = v;
    mLumiScale = mMeanLumi ? mInstLumi / mMeanLumi : 0.f;
  }

  GPUd() float getInstLumi() const { return mInstLumi; }
  GPUd() float getMeanLumi() const { return mMeanLumi; }
  GPUd() float getLumiScale() const { return mLumiScale; }

  bool isUpdated() const { return mUpdatedFlags != 0; }
  bool isUpdatedMap() const { return (mUpdatedFlags & UpdateFlags::MapBit) == 0; }
  bool isUpdatedMapRef() const { return (mUpdatedFlags & UpdateFlags::MapRefBit) == 0; }
  bool isUpdatedLumi() const { return (mUpdatedFlags & UpdateFlags::LumiBit) == 0; }
  void setUpdatedMap() { mUpdatedFlags |= UpdateFlags::MapBit; }
  void setUpdatedMapRef() { mUpdatedFlags |= UpdateFlags::MapRefBit; }
  void setUpdatedLumi() { mUpdatedFlags |= UpdateFlags::LumiBit; }

#ifndef GPUCA_GPUCODE_DEVICE
  void setCorrMap(std::unique_ptr<o2::gpu::TPCFastTransform>&& m);
  void setCorrMapRef(std::unique_ptr<o2::gpu::TPCFastTransform>&& m);
  void setOwner(bool v) { mOwner = v; }
  void acknowledgeUpdate() { mUpdatedFlags = 0; }
#endif

 protected:
  enum UpdateFlags { MapBit = 0x1,
                     MapRefBit = 0x2,
                     LumiBit = 0x4 };
  bool mOwner = false; // is content of pointers owned by the helper
  int mUpdatedFlags = 0;
  float mInstLumi = 0.;                            // instanteneous luminosity (a.u)
  float mMeanLumi = 0.;                            // mean luminosity of the map (a.u)
  float mLumiScale = 0.;                           // precalculated mInstLumi/mMeanLumi
  o2::gpu::TPCFastTransform* mCorrMap{nullptr};    // current transform
  o2::gpu::TPCFastTransform* mCorrMapRef{nullptr}; // reference transform
#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(CorrectionMapsHelper, 1);
#endif
};

} // namespace o2::gpu

#endif
