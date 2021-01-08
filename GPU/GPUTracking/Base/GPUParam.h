// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace base
{
class Propagator;
} // namespace base
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUSettingsRec;
struct GPUSettingsEvent;
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

  GPUTPCGeometry tpcGeometry;              // TPC Geometry
  GPUTPCGMPolynomialField polynomialField; // Polynomial approx. of magnetic field for TPC GM

  GPUParamSlice SliceParam[GPUCA_NSLICES];

 protected:
  float ParamRMS0[2][3][4];  // cluster shape parameterization coeficients
  float ParamS0Par[2][3][6]; // cluster error parameterization coeficients
};
} // namespace internal

#if !(defined(__CINT__) || defined(__ROOTCINT__)) || defined(__CLING__) // Hide from ROOT 5 CINT since it triggers a CINT but
MEM_CLASS_PRE()
struct GPUParam : public internal::GPUParam_t<GPUSettingsRec, GPUSettingsParam> {

#ifndef GPUCA_GPUCODE
  void SetDefaults(float solenoidBz);
  void SetDefaults(const GPUSettingsEvent* e, const GPUSettingsRec* r = nullptr, const GPUSettingsProcessing* p = nullptr, const GPURecoStepConfiguration* w = nullptr);
  void UpdateEventSettings(const GPUSettingsEvent* e, const GPUSettingsProcessing* p = nullptr);
  void LoadClusterErrors(bool Print = 0);
  o2::base::Propagator* GetDefaultO2Propagator(bool useGPUField = false) const;
#endif

  GPUd() float Alpha(int iSlice) const
  {
    if (iSlice >= GPUCA_NSLICES / 2) {
      iSlice -= GPUCA_NSLICES / 2;
    }
    if (iSlice >= GPUCA_NSLICES / 4) {
      iSlice -= GPUCA_NSLICES / 2;
    }
    return 0.174533f + par.DAlpha * iSlice;
  }
  GPUd() float GetClusterRMS(int yz, int type, float z, float angle2) const;
  GPUd() void GetClusterRMS2(int row, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const;

  GPUd() float GetClusterError2(int yz, int type, float z, float angle2) const;
  GPUd() void GetClusterErrors2(int row, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const;
  GPUd() void UpdateClusterError2ByState(short clusterState, float& ErrY2, float& ErrZ2) const;

  GPUd() void Slice2Global(int iSlice, float x, float y, float z, float* X, float* Y, float* Z) const;
  GPUd() void Global2Slice(int iSlice, float x, float y, float z, float* X, float* Y, float* Z) const;
};
#endif

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
