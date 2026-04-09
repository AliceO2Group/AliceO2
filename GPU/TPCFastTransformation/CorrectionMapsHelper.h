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
/// \author ruben.shahoian@cern.ch matthias.kleiner@cern.ch

#ifndef TPC_CORRECTION_MAPS_HELPER_H_
#define TPC_CORRECTION_MAPS_HELPER_H_

#include "TPCFastTransform.h"
#include "CorrectionMapsTypes.h"

namespace o2::gpu
{

class CorrectionMapsHelper
{
 public:
  CorrectionMapsHelper() = default;
  ~CorrectionMapsHelper() { clear(); }
  CorrectionMapsHelper(const CorrectionMapsHelper&) = delete;
  void updateLumiScale(bool report = false);
  void clear();

  const o2::gpu::TPCFastTransform* getCorrMap() const { return mCorrMap; }
  const o2::gpu::TPCFastTransform* getCorrMapRef() const { return mCorrMapRef; }
  const o2::gpu::TPCFastTransform* getCorrMapMShape() const { return mCorrMapMShape.get(); }

  void setCorrMap(o2::gpu::TPCFastTransform* m) { mCorrMap = m; }
  void setCorrMapRef(o2::gpu::TPCFastTransform* m) { mCorrMapRef = m; }
  void setCorrMapMShape(std::unique_ptr<o2::gpu::TPCFastTransform>&& m);

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

  void setMeanLumiRef(float v, bool report = false)
  {
    if (v != mMeanLumiRef) {
      mMeanLumiRef = v;
      updateLumiScale(report);
    }
  }

  void setLumiScaleMode(tpc::LumiScaleMode v)
  {
    if (v != mLumiScaleMode) {
      mLumiScaleMode = v;
      updateLumiScale(false);
    }
  }

  void setCheckCTPIDCConsistency(bool v) { mCheckCTPIDCConsistency = v; }
  bool getCheckCTPIDCConsistency() const { return mCheckCTPIDCConsistency; }

  float getInstLumiCTP() const { return mInstLumiCTP; }
  float getInstLumi() const { return mInstLumi; }
  float getMeanLumi() const { return mMeanLumi; }
  float getMeanLumiRef() const { return mMeanLumiRef; }

  float getLumiScale() const { return mLumiScale; }
  tpc::LumiScaleMode getLumiScaleMode() const { return mLumiScaleMode; }

  bool isUpdated() const { return mUpdatedFlags != 0; }
  bool isUpdatedMap() const { return (mUpdatedFlags & UpdateFlags::MapBit) != 0; }
  bool isUpdatedMapRef() const { return (mUpdatedFlags & UpdateFlags::MapRefBit) != 0; }
  bool isUpdatedMapMShape() const { return (mUpdatedFlags & UpdateFlags::MapMShapeBit) != 0; }
  bool isUpdatedLumi() const { return (mUpdatedFlags & UpdateFlags::LumiBit) != 0; }
  void setUpdatedMap() { mUpdatedFlags |= UpdateFlags::MapBit; }
  void setUpdatedMapRef() { mUpdatedFlags |= UpdateFlags::MapRefBit; }
  void setUpdatedMapMShape() { mUpdatedFlags |= UpdateFlags::MapMShapeBit; }
  void setUpdatedLumi() { mUpdatedFlags |= UpdateFlags::LumiBit; }
  void acknowledgeUpdate() { mUpdatedFlags = 0; }
  void setLumiCTPAvailable(bool v) { mLumiCTPAvailable = v; }
  bool getLumiCTPAvailable() const { return mLumiCTPAvailable; }
  void setLumiScaleType(tpc::LumiScaleType v) { mLumiScaleType = v; }
  tpc::LumiScaleType getLumiScaleType() const { return mLumiScaleType; }
  void enableMShapeCorrection(bool v) { mEnableMShape = v; }
  bool getUseMShapeCorrection() const { return mEnableMShape; }
  bool canUseCorrections() const { return mMeanLumi >= 0.; }
  void setMeanLumiOverride(float f) { mMeanLumiOverride = f; }
  void setMeanLumiRefOverride(float f) { mMeanLumiRefOverride = f; }
  float getMeanLumiOverride() const { return mMeanLumiOverride; }
  float getMeanLumiRefOverride() const { return mMeanLumiRefOverride; }

  void setInstCTPLumiOverride(float f) { mInstCTPLumiOverride = f; }
  float getInstCTPLumiOverride() const { return mInstCTPLumiOverride; }

  int32_t getUpdateFlags() const { return mUpdatedFlags; }

  /// return returns if the correction map for the M-shape correction is a dummy spline object
  bool isCorrMapMShapeDummy() const
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
  bool mLumiCTPAvailable = false; // is CTP Lumi available
  // these 2 are global options, must be set by the workflow global options
  tpc::LumiScaleType mLumiScaleType = tpc::LumiScaleType::Unset; // use CTP Lumi (1) or TPCScaler (2) for the correction scaling, 0 - no scaling
  tpc::LumiScaleMode mLumiScaleMode = tpc::LumiScaleMode::Unset; // scaling-mode of the correction maps: 0 = linear scaling, 1 = using the derivative map, 2 = using the derivative map for MC (i.e. only apply the scaled derivative on top of the reference map)
  int32_t mUpdatedFlags = 0;
  float mInstLumiCTP = 0.;                                            // instanteneous luminosity from CTP (a.u)
  float mInstLumi = 0.;                                               // instanteneous luminosity (a.u) used for TPC corrections scaling
  float mMeanLumi = 0.;                                               // mean luminosity of the map (a.u) used for TPC corrections scaling
  float mMeanLumiRef = 0.;                                            // mean luminosity of the ref map (a.u) used for TPC corrections scaling reference
  float mLumiScale = 0.;                                              // precalculated mInstLumi/mMeanLumi
  float mMeanLumiOverride = -1.f;                                     // optional value to override mean lumi
  float mMeanLumiRefOverride = -1.f;                                  // optional value to override ref mean lumi
  float mInstCTPLumiOverride = -1.f;                                  // optional value to override inst lumi from CTP
  bool mEnableMShape = false;                                         ///< use v shape correction
  bool mCheckCTPIDCConsistency{true};                                 // check of selected CTP or IDC scaling source being consistent with the map
  o2::gpu::TPCFastTransform* mCorrMap{nullptr};                       // current transform
  o2::gpu::TPCFastTransform* mCorrMapRef{nullptr};                    // reference transform
  std::unique_ptr<o2::gpu::TPCFastTransform> mCorrMapMShape{nullptr}; // correction map for M-shape distortions on A-side
  ClassDefNV(CorrectionMapsHelper, 6);
};

} // namespace o2::gpu

#endif
