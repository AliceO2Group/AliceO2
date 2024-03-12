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
#include "GPUTPCGeometry.h"
#include "GPUTPCGMPolynomialField.h"

#if !defined(GPUCA_GPUCODE) && defined(GPUCA_NOCOMPAT)
namespace o2::base
{
template <typename>
class PropagatorImpl;
using Propagator = PropagatorImpl<float>;
} // namespace o2::base
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUSettingsRec;
struct GPUSettingsGTP;
struct GPURecoStepConfiguration;

struct GPUParamSlice {
  float Alpha;              // slice angle
  float CosAlpha, SinAlpha; // sign and cosine of the slice angle
  float AngleMin, AngleMax; // minimal and maximal angle
  float ZMin, ZMax;         // slice Z range
};

namespace internal
{
template <class T, class S>
struct GPUParam_t {
  T rec;
  S par;

  float bzkG;
  float constBz;
  float qptB5Scaler;

  GPUTPCGeometry tpcGeometry;                       // TPC Geometry
  GPUTPCGMPolynomialField polynomialField;          // Polynomial approx. of magnetic field for TPC GM
  const unsigned int* occupancyMap;                 // Ptr to TPC occupancy map

  GPUParamSlice SliceParam[GPUCA_NSLICES];

 protected:
#ifdef GPUCA_TPC_GEOMETRY_O2
  float ParamErrors[2][4][4]; // cluster error parameterization used during seeding and fit
#else
  float ParamErrorsSeeding0[2][3][4]; // cluster error parameterization used during seeding
  float ParamS0Par[2][3][6];          // cluster error parameterization used during track fit
#endif
};
} // namespace internal

#if !(defined(__CINT__) || defined(__ROOTCINT__)) || defined(__CLING__) // Hide from ROOT 5 CINT
MEM_CLASS_PRE()
struct GPUParam : public internal::GPUParam_t<GPUSettingsRec, GPUSettingsParam> {

#ifndef GPUCA_GPUCODE
  void SetDefaults(float solenoidBz);
  void SetDefaults(const GPUSettingsGRP* g, const GPUSettingsRec* r = nullptr, const GPUSettingsProcessing* p = nullptr, const GPURecoStepConfiguration* w = nullptr);
  void UpdateSettings(const GPUSettingsGRP* g, const GPUSettingsProcessing* p = nullptr, const GPURecoStepConfiguration* w = nullptr);
  void LoadClusterErrors(bool Print = 0);
  o2::base::Propagator* GetDefaultO2Propagator(bool useGPUField = false) const;
  void UpdateRun3ClusterErrors(const float* yErrorParam, const float* zErrorParam);
#endif

  GPUd() float Alpha(int iSlice) const
  {
    if (iSlice >= GPUCA_NSLICES / 2) {
      iSlice -= GPUCA_NSLICES / 2;
    }
    if (iSlice >= GPUCA_NSLICES / 4) {
      iSlice -= GPUCA_NSLICES / 2;
    }
    return 0.174533f + par.dAlpha * iSlice;
  }
  GPUd() float GetClusterErrorSeeding(int yz, int type, float zDiff, float angle2) const;
  GPUd() void GetClusterErrorsSeeding2(char sector, int row, float z, float sinPhi, float DzDs, float time, float avgInvCharge, float invCharge, float& ErrY2, float& ErrZ2) const;
  GPUd() float GetSystematicClusterErrorIFC2(float x, float y, float z, bool sideC) const;

  GPUd() float GetClusterError2(int yz, int type, float zDiff, float angle2, float scaledMult, float scaledAvgInvCharge, float scaledInvCharge) const;
  GPUd() void GetClusterErrors2(char sector, int row, float z, float sinPhi, float DzDs, float time, float avgInvCharge, float invCharge, float& ErrY2, float& ErrZ2) const;
  GPUd() void UpdateClusterError2ByState(short clusterState, float& ErrY2, float& ErrZ2) const;
  GPUd() float GetScaledMult(float time) const;

  GPUd() void Slice2Global(int iSlice, float x, float y, float z, float* X, float* Y, float* Z) const;
  GPUd() void Global2Slice(int iSlice, float x, float y, float z, float* X, float* Y, float* Z) const;

  GPUd() bool rejectEdgeClusterByY(float uncorrectedY, int iRow) const;
};
#endif

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
