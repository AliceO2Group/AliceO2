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
#include "GPUCommonDef.h"
#include "TPCFastTransform.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class CorrectionMapsHelper
{
 public:
  CorrectionMapsHelper() = default;
  ~CorrectionMapsHelper() { clear(); }
  CorrectionMapsHelper(const CorrectionMapsHelper&) = delete;
  void updateLumiScale(bool report = false);
  void clear();

  GPUd() void Transform(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime = 0) const
  {
    mCorrMap->Transform(slice, row, pad, time, x, y, z, vertexTime, mCorrMapRef, mCorrMapMShape, mLumiScale, 1, mLumiScaleMode);
  }

  GPUd() void TransformXYZ(int slice, int row, float& x, float& y, float& z) const
  {
    mCorrMap->TransformXYZ(slice, row, x, y, z, mCorrMapRef, mCorrMapMShape, mLumiScale, 1, mLumiScaleMode);
  }

  GPUd() void InverseTransformYZtoX(int slice, int row, float y, float z, float& x) const
  {
    mCorrMap->InverseTransformYZtoX(slice, row, y, z, x, mCorrMapRef, mCorrMapMShape, (mScaleInverse ? mLumiScale : 0), (mScaleInverse ? 1 : 0), mLumiScaleMode);
  }

  GPUd() void InverseTransformYZtoNominalYZ(int slice, int row, float y, float z, float& ny, float& nz) const
  {
    mCorrMap->InverseTransformYZtoNominalYZ(slice, row, y, z, ny, nz, mCorrMapRef, mCorrMapMShape, (mScaleInverse ? mLumiScale : 0), (mScaleInverse ? 1 : 0), mLumiScaleMode);
  }

  GPUd() const GPUCA_NAMESPACE::gpu::TPCFastTransform* getCorrMap() const { return mCorrMap; }
  GPUd() const GPUCA_NAMESPACE::gpu::TPCFastTransform* getCorrMapRef() const { return mCorrMapRef; }
  GPUd() const GPUCA_NAMESPACE::gpu::TPCFastTransform* getCorrMapMShape() const { return mCorrMapMShape; }

  bool getOwner() const { return mOwner; }

  void setCorrMap(GPUCA_NAMESPACE::gpu::TPCFastTransform* m);
  void setCorrMapRef(GPUCA_NAMESPACE::gpu::TPCFastTransform* m);
  void setCorrMapMShape(GPUCA_NAMESPACE::gpu::TPCFastTransform* m);
  void reportScaling();
  void setInstLumiCTP(float v)
  {
    if (v != mInstLumiCTP) {
      mInstLumiCTP = v;
    }
  }

  void setInstLumi(float v, bool report = false)
  {
    if (v != mInstLumi) {
      mInstLumi = v;
      updateLumiScale(report);
    }
  }

  void setMeanLumi(float v, bool report = false)
  {
    if (v != mMeanLumi) {
      mMeanLumi = v;
      updateLumiScale(report);
    }
  }

  void setMeanLumiRef(float v)
  {
    if (v != mMeanLumi) {
      mMeanLumiRef = v;
    }
  }

  void setLumiScaleMode(int v)
  {
    if (v != mLumiScaleMode) {
      mLumiScaleMode = v;
      updateLumiScale(false);
    }
  }

  GPUd() float getInstLumiCTP() const { return mInstLumiCTP; }
  GPUd() float getInstLumi() const { return mInstLumi; }
  GPUd() float getMeanLumi() const { return mMeanLumi; }
  GPUd() float getMeanLumiRef() const { return mMeanLumiRef; }

  GPUd() float getLumiScale() const { return mLumiScale; }
  GPUd() int getLumiScaleMode() const { return mLumiScaleMode; }

  bool isUpdated() const { return mUpdatedFlags != 0; }
  bool isUpdatedMap() const { return (mUpdatedFlags & UpdateFlags::MapBit) != 0; }
  bool isUpdatedMapRef() const { return (mUpdatedFlags & UpdateFlags::MapRefBit) != 0; }
  bool isUpdatedMapMShape() const { return (mUpdatedFlags & UpdateFlags::MapMShapeBit) != 0; }
  bool isUpdatedLumi() const { return (mUpdatedFlags & UpdateFlags::LumiBit) != 0; }
  void setUpdatedMap() { mUpdatedFlags |= UpdateFlags::MapBit; }
  void setUpdatedMapRef() { mUpdatedFlags |= UpdateFlags::MapRefBit; }
  void setUpdatedMapMShape() { mUpdatedFlags |= UpdateFlags::MapMShapeBit; }
  void setUpdatedLumi() { mUpdatedFlags |= UpdateFlags::LumiBit; }

#if !defined(GPUCA_GPUCODE_DEVICE) && defined(GPUCA_NOCOMPAT)
  void setCorrMap(std::unique_ptr<GPUCA_NAMESPACE::gpu::TPCFastTransform>&& m);
  void setCorrMapRef(std::unique_ptr<GPUCA_NAMESPACE::gpu::TPCFastTransform>&& m);
  void setCorrMapMShape(std::unique_ptr<GPUCA_NAMESPACE::gpu::TPCFastTransform>&& m);
#endif
  void setOwner(bool v);
  void acknowledgeUpdate() { mUpdatedFlags = 0; }
  void setLumiCTPAvailable(bool v) { mLumiCTPAvailable = v; }
  bool getLumiCTPAvailable() const { return mLumiCTPAvailable; }
  void setLumiScaleType(int v) { mLumiScaleType = v; }
  int getLumiScaleType() const { return mLumiScaleType; }
  void enableMShapeCorrection(bool v) { mEnableMShape = v; }
  bool getUseMShapeCorrection() const { return mEnableMShape; }
  bool canUseCorrections() const { return mMeanLumi >= 0.; }
  void setMeanLumiOverride(float f) { mMeanLumiOverride = f; }
  void setMeanLumiRefOverride(float f) { mMeanLumiRefOverride = f; }
  float getMeanLumiOverride() const { return mMeanLumiOverride; }
  float getMeanLumiRefOverride() const { return mMeanLumiRefOverride; }

  void setInstCTPLumiOverride(float f) { mInstCTPLumiOverride = f; }
  float getInstCTPLumiOverride() const { return mInstCTPLumiOverride; }

  int getUpdateFlags() const { return mUpdatedFlags; }

  bool getScaleInverse() const { return mScaleInverse; }

  /// return returns if the correction map for the M-shape correction is a dummy spline object
  GPUd() bool isCorrMapMShapeDummy() const
  {
    if (mCorrMapMShape) {
      // just check for the first spline the number of knots which are 4 in case of default spline object
      return mCorrMapMShape->getCorrection().getSpline(0, 0).getNumberOfKnots() == 4;
    }
    return true;
  }

 protected:
  enum UpdateFlags { MapBit = 0x1,
                     MapRefBit = 0x2,
                     LumiBit = 0x4,
                     MapMShapeBit = 0x10 };
  bool mOwner = false; // is content of pointers owned by the helper
  bool mLumiCTPAvailable = false; // is CTP Lumi available
  // these 2 are global options, must be set by the workflow global options
  int mLumiScaleType = -1; // use CTP Lumi (1) or TPCScaler (2) for the correction scaling, 0 - no scaling
  int mLumiScaleMode = -1; // scaling-mode of the correciton maps
  int mUpdatedFlags = 0;
  float mInstLumiCTP = 0.;                                         // instanteneous luminosity from CTP (a.u)
  float mInstLumi = 0.;                                            // instanteneous luminosity (a.u) used for TPC corrections scaling
  float mMeanLumi = 0.;                                            // mean luminosity of the map (a.u) used for TPC corrections scaling
  float mMeanLumiRef = 0.;                                         // mean luminosity of the ref map (a.u) used for TPC corrections scaling reference
  float mLumiScale = 0.;                                           // precalculated mInstLumi/mMeanLumi
  float mMeanLumiOverride = -1.f;                                  // optional value to override mean lumi
  float mMeanLumiRefOverride = -1.f;                               // optional value to override ref mean lumi
  float mInstCTPLumiOverride = -1.f;                               // optional value to override inst lumi from CTP
  bool mEnableMShape = false;                                      ///< use v shape correction
  bool mScaleInverse{false};                                       // if set to false the inverse correction is already scaled and will not scaled again
  GPUCA_NAMESPACE::gpu::TPCFastTransform* mCorrMap{nullptr};       // current transform
  GPUCA_NAMESPACE::gpu::TPCFastTransform* mCorrMapRef{nullptr};    // reference transform
  GPUCA_NAMESPACE::gpu::TPCFastTransform* mCorrMapMShape{nullptr}; // correction map for v-shape distortions on A-side
#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(CorrectionMapsHelper, 6);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
