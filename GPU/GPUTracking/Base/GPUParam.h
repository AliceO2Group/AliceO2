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

MEM_CLASS_PRE()
struct GPUParam {
 public:
  GPUSettingsRec rec;

  float DAlpha;           // angular size
  float RMin, RMax;       // slice R range
  float ErrX, ErrY, ErrZ; // default cluster errors
  float PadPitch;         // pad pitch
  float BzkG;             // constant magnetic field value in kG
  float ConstBz;          // constant magnetic field value in kG*clight

  char AssumeConstantBz;                   // Assume a constant magnetic field
  char ToyMCEventsFlag;                    // events were build with home-made event generator
  char ContinuousTracking;                 // Continuous tracking, estimate bz and errors for abs(z) = 125cm during seeding
  char resetTimers;                        // Reset benchmark timers before event processing
  char dodEdx;                             // Do dEdx computation
  int debugLevel;                          // Debug level
  int continuousMaxTimeBin;                // Max time bin for continuous tracking
  GPUTPCGeometry tpcGeometry;              // TPC Geometry
  GPUTPCGMPolynomialField polynomialField; // Polynomial approx. of magnetic field for TPC GM

  GPUParamSlice SliceParam[GPUCA_NSLICES];

#ifndef GPUCA_GPUCODE
  void SetDefaults(float solenoidBz);
  void SetDefaults(const GPUSettingsEvent* e, const GPUSettingsRec* r = nullptr, const GPUSettingsDeviceProcessing* p = nullptr, const GPURecoStepConfiguration* w = nullptr);
  void UpdateEventSettings(const GPUSettingsEvent* e, const GPUSettingsDeviceProcessing* p = nullptr);
  void LoadClusterErrors(bool Print = 0);
#endif

  GPUd() float Alpha(int iSlice) const
  {
    if (iSlice >= GPUCA_NSLICES / 2) {
      iSlice -= GPUCA_NSLICES / 2;
    }
    if (iSlice >= GPUCA_NSLICES / 4) {
      iSlice -= GPUCA_NSLICES / 2;
    }
    return 0.174533f + DAlpha * iSlice;
  }
  GPUd() float GetClusterRMS(int yz, int type, float z, float angle2) const;
  GPUd() void GetClusterRMS2(int row, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const;

  GPUd() float GetClusterError2(int yz, int type, float z, float angle2) const;
  GPUd() void GetClusterErrors2(int row, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const;

  void Slice2Global(int iSlice, float x, float y, float z, float* X, float* Y, float* Z) const;
  void Global2Slice(int iSlice, float x, float y, float z, float* X, float* Y, float* Z) const;

 protected:
  float ParamRMS0[2][3][4];  // cluster shape parameterization coeficients
  float ParamS0Par[2][3][6]; // cluster error parameterization coeficients
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
