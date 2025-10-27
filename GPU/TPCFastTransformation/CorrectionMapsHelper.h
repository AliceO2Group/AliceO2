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
#include <vector>
#endif
#include "GPUCommonDef.h"
#include "TPCFastTransformPOD.h"

namespace o2
{

namespace framework
{
class ConfigParamRegistry;
} // namespace framework

namespace tpc
{
enum class LumiScaleType : int32_t {
  Unset = -1, ///< init value
  NoScaling = 0, ///< no scaling, use map as is
  CTPLumi = 1, ///< use CTP luminosity for scaling
  TPCScaler = 2 ///< use TPC scaler for scaling
};

enum class LumiScaleMode : int32_t {
    Unset = -1, ///< init value
    Linear = 0, ///< map(lumi) = (mean_map - referenceMap) * lumiScale + referenceMap
    DerivativeMap = 1, ///< map(lumi) = mean_map + lumiScale * (derivativeMap) where derivativeMap = (mean_map_A - mean_map_B)
    DerivativeMapMC = 2  ///< same DerivativeMap, but for MC
};

struct CorrectionMapsLoaderGloOpts {
  LumiScaleType lumiType = LumiScaleType::Unset; ///< what estimator to used for corrections scaling: 0: no scaling, 1: CTP, 2: IDC
  LumiScaleMode lumiMode = LumiScaleMode::Unset; ///< what corrections method to use: 0: classical scaling, 1: Using of the derivative map, 2: Using of the derivative map for MC
  bool enableMShapeCorrection = false;
  bool requestCTPLumi = true; ///< request CTP Lumi regardless of what is used for corrections scaling
  bool checkCTPIDCconsistency = true; ///< check the selected CTP or IDC scaling source being consistent with mean scaler of the map
};
}

namespace gpu
{

class CorrectionMapsHelper
{
 public:
  CorrectionMapsHelper() = default;
  ~CorrectionMapsHelper() { clear(); }
  CorrectionMapsHelper(const CorrectionMapsHelper&) = delete;
  static tpc::CorrectionMapsLoaderGloOpts parseGlobalOptions(const o2::framework::ConfigParamRegistry& opts);
  void setUpdatedMap() { mUpdated = true; }
  void clear()
  {
    mCorrMap = nullptr;
#if !defined(GPUCA_GPUCODE_DEVICE)
    mCorrMapBuffer.clear();
#endif
  }

  GPUd() void Transform(int32_t slice, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime = 0) const
  {
    mCorrMap->Transform(slice, row, pad, time, x, y, z, vertexTime);
  }

  GPUd() void TransformXYZ(int32_t slice, int32_t row, float& x, float& y, float& z) const
  {
    mCorrMap->TransformXYZ(slice, row, x, y, z);
  }

  GPUd() void InverseTransformYZtoX(int32_t slice, int32_t row, float y, float z, float& x) const
  {
    mCorrMap->InverseTransformYZtoX(slice, row, y, z, x);
  }

  GPUd() void InverseTransformYZtoNominalYZ(int32_t slice, int32_t row, float y, float z, float& ny, float& nz) const
  {
    mCorrMap->InverseTransformYZtoNominalYZ(slice, row, y, z, ny, nz);
  }

  GPUd() const o2::gpu::TPCFastTransformPOD* getCorrMap() const { return mCorrMap; }

  float getInstLumiCTP() const { return mInstLumiCTP; }

  bool isUpdated() const { return mUpdated; }
  void acknowledgeUpdate() { mUpdated = false; }
  void setCorrMap(const o2::gpu::TPCFastTransformPOD* m); // always non-owning
#if !defined(GPUCA_GPUCODE_DEVICE)
  void setCorrMap(std::vector<char>&& buffer); // owning
#endif

 protected:
  bool mUpdated = false;                                 // flag indicating whether the map was updated
  float mInstLumiCTP{-1.f};                              // current CTP luminosity - used for track covariance tuning in downstream devices
  const o2::gpu::TPCFastTransformPOD* mCorrMap{nullptr}; // current transform
#if !defined(GPUCA_GPUCODE_DEVICE)
  std::vector<char> mCorrMapBuffer;
#endif
  ClassDefNV(CorrectionMapsHelper, 1);
};

} // namespace gpu
} // namespace o2

#endif
