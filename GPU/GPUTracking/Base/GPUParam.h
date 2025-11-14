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

/// \file GPUParam.h
/// \author David Rohr, Sergey Gorbunov

#ifndef GPUPARAM_H
#define GPUPARAM_H

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUDef.h"
#include "GPUSettings.h"
#include "GPUTPCGMPolynomialField.h"

#if !defined(GPUCA_GPUCODE)
namespace o2::base
{
template <typename>
class PropagatorImpl;
using Propagator = PropagatorImpl<float>;
} // namespace o2::base
#endif

namespace o2::gpu
{
struct GPUSettingsRec;
struct GPUSettingsGTP;
struct GPURecoStepConfiguration;

struct GPUParamSector {
  float Alpha;              // sector angle
  float CosAlpha, SinAlpha; // sign and cosine of the sector angle
  float AngleMin, AngleMax; // minimal and maximal angle
  float ZMin, ZMax;         // sector Z range
};

namespace internal
{
template <class T, class S>
struct GPUParam_t {
  static constexpr float dAlpha = 0.349066f;

  T rec;
  S par;

  float bzkG;
  float bzCLight;
  float qptB5Scaler;

  int8_t dodEdxEnabled;
  int32_t continuousMaxTimeBin;
  int32_t tpcCutTimeBin;

  GPUTPCGMPolynomialField polynomialField; // Polynomial approx. of magnetic field for TPC GM
  const uint32_t* occupancyMap;            // Ptr to TPC occupancy map
  uint32_t occupancyTotal;                 // Total occupancy in the TPC (nCl / nHbf)
  uint32_t occupancyMapSize;               // Size of occupancy map

  GPUParamSector SectorParam[GPUCA_NSECTORS];

 protected:
#ifdef GPUCA_TPC_GEOMETRY_O2
  float ParamErrors[2][4][4]; // cluster error parameterization used during seeding and fit
#else
  float ParamErrorsSeeding0[2][3][4]; // cluster error parameterization used during seeding
  float ParamS0Par[2][3][6];          // cluster error parameterization used during track fit
#endif
};
} // namespace internal

struct GPUParam : public internal::GPUParam_t<GPUSettingsRec, GPUSettingsParam> {

#ifndef GPUCA_GPUCODE
  void SetDefaults(float solenoidBz, bool assumeConstantBz);
  void SetDefaults(const GPUSettingsGRP* g, const GPUSettingsRec* r = nullptr, const GPUSettingsProcessing* p = nullptr, const GPURecoStepConfiguration* w = nullptr);
  void UpdateSettings(const GPUSettingsGRP* g, const GPUSettingsProcessing* p = nullptr, const GPURecoStepConfiguration* w = nullptr, const GPUSettingsRecDynamic* d = nullptr);
  void UpdateBzOnly(float newSolenoidBz, bool assumeConstantBz);
  void UpdateRun3ClusterErrors(const float* yErrorParam, const float* zErrorParam);
#endif

  GPUd() float Alpha(int32_t iSector) const
  {
    if (iSector >= GPUCA_NSECTORS / 2) {
      iSector -= GPUCA_NSECTORS / 2;
    }
    if (iSector >= GPUCA_NSECTORS / 4) {
      iSector -= GPUCA_NSECTORS / 2;
    }
    return 0.174533f + dAlpha * iSector;
  }
  GPUd() float GetClusterErrorSeeding(int32_t yz, int32_t type, float zDiff, float angle2, float unscaledMult) const;
  GPUd() void GetClusterErrorsSeeding2(uint8_t sector, int32_t row, float z, float sinPhi, float DzDs, float time, float& ErrY2, float& ErrZ2) const;
  GPUd() float GetSystematicClusterErrorIFC2(float trackX, float trackY, float z, bool sideC) const;
  GPUd() float GetSystematicClusterErrorC122(float trackX, float trackY, uint8_t sector) const;

  GPUd() float GetClusterError2(int32_t yz, int32_t type, float zDiff, float angle2, float unscaledMult, float scaledAvgInvCharge, float scaledInvCharge) const;
  GPUd() void GetClusterErrors2(uint8_t sector, int32_t row, float z, float sinPhi, float DzDs, float time, float avgInvCharge, float invCharge, float& ErrY2, float& ErrZ2) const;
  GPUd() void UpdateClusterError2ByState(int16_t clusterState, float& ErrY2, float& ErrZ2) const;
  GPUd() float GetUnscaledMult(float time) const;

  GPUd() void Sector2Global(int32_t iSector, float x, float y, float z, float* X, float* Y, float* Z) const;
  GPUd() void Global2Sector(int32_t iSector, float x, float y, float z, float* X, float* Y, float* Z) const;

  GPUd() bool rejectEdgeClusterByY(float uncorrectedY, int32_t iRow, float trackSigmaY) const;
};

} // namespace o2::gpu

#endif
