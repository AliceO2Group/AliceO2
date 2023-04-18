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

/// \file GPUTPCCompressionTrackModel.h
/// \author David Rohr

#ifndef GPUTPCCOMPRESSIONTRACKMODEL_H
#define GPUTPCCOMPRESSIONTRACKMODEL_H

// For debugging purposes, we provide means to use other track models
// #define GPUCA_COMPRESSION_TRACK_MODEL_MERGER
// #define GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER

#include "GPUDef.h"

#ifdef GPUCA_COMPRESSION_TRACK_MODEL_MERGER
#include "GPUTPCGMPropagator.h"
#include "GPUTPCGMTrackParam.h"

#elif defined(GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER)
#include "GPUTPCTrackParam.h"

#else // Default internal track model for compression
#endif

namespace GPUCA_NAMESPACE::gpu
{
// ATTENTION! This track model is used for the data compression.
// Changes to the propagation and fit will prevent the decompression of data
// encoded with the old version!!!

struct GPUParam;

constexpr float MaxSinPhi = 0.999f;

class GPUTPCCompressionTrackModel
{
 public:
  GPUd() void Init(float x, float y, float z, float alpha, unsigned char qPt, const GPUParam& proc);
  GPUd() int Propagate(float x, float alpha);
  GPUd() int Filter(float y, float z, int iRow);
  GPUd() int Mirror();

#if defined(GPUCA_COMPRESSION_TRACK_MODEL_MERGER) || defined(GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER)
  GPUd() float X() const
  {
    return mTrk.GetX();
  }
  GPUd() float Y() const { return mTrk.GetY(); }
  GPUd() float Z() const { return mTrk.GetZ(); }
  GPUd() float SinPhi() const { return mTrk.GetSinPhi(); }
  GPUd() float DzDs() const { return mTrk.GetDzDs(); }
  GPUd() float QPt() const { return mTrk.GetQPt(); }

#else // Default internal track model for compression

  struct PhysicalTrackModel { // see GPUTPCGMPhysicalTrackModel
    // physical parameters of the trajectory

    float x = 0.f;    // X
    float y = 0.f;    // Y
    float z = 0.f;    // Z
    float px = 1.e4f; // Px, >0
    float py = 0.f;   // Py
    float pz = 0.f;   // Pz
    float q = 1.f;    // charge, +-1

    // some additional variables needed for GMTrackParam transport

    float sinphi = 0.f; // SinPhi = Py/Pt
    float cosphi = 1.f; // CosPhi = abs(Px)/Pt
    float secphi = 1.f; // 1/cos(phi) = Pt/abs(Px)
    float dzds = 0.f;   // DzDs = Pz/Pt
    float dlds = 1.f;   // DlDs = P/Pt
    float qpt = 0.f;    // QPt = q/Pt
    float p = 1.e4f;    // momentum
    float pt = 1.e4f;   // Pt momentum
  };

  GPUd() float Y() const { return mP[0]; }
  GPUd() float Z() const { return mP[1]; }

  // helper functions for standalone propagation and update methods
  GPUd() void updatePhysicalTrackValues(PhysicalTrackModel& trk);
  GPUd() void changeDirection();
  GPUd() int rotateToAlpha(float newAlpha);
  GPUd() int propagateToXBzLightNoUpdate(PhysicalTrackModel& t, float x, float Bz, float& dLp);
  GPUd() bool setDirectionAlongX(PhysicalTrackModel& t);
  GPUd() int followLinearization(const PhysicalTrackModel& t0e, float Bz, float dLp);
  GPUd() void calculateMaterialCorrection();
  GPUd() float approximateBetheBloch(float beta2);
  GPUd() void getClusterErrors2(int iRow, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const;
  GPUd() void resetCovariance();

#endif

 protected:
#ifdef GPUCA_COMPRESSION_TRACK_MODEL_MERGER
  GPUTPCGMPropagator mProp;
  GPUTPCGMTrackParam mTrk;
  const GPUParam* mParam;

#elif defined(GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER)
  GPUTPCTrackParam mTrk;
  float mAlpha;
  const GPUParam* mParam;

#else // Default internal track model for compression

  struct MaterialCorrection {
    GPUhd() MaterialCorrection() : radLen(28811.7f), rho(1.025e-3f), radLenInv(1.f / radLen), DLMax(0.f), EP2(0.f), sigmadE2(0.f), k22(0.f), k33(0.f), k43(0.f), k44(0.f) {}

    float radLen;                                              // [cm]
    float rho;                                                 // [g/cm^3]
    float radLenInv, DLMax, EP2, sigmadE2, k22, k33, k43, k44; // precalculated values for MS and EnergyLoss correction
  };

  // default TPC cluster error parameterization taken from GPUParam.cxx
  // clang-format off
  const float mParamErrors0[2][3][4] =
  {
    { { 4.17516864836e-02, 1.87623649254e-04, 5.63788712025e-02, 5.38373768330e-01, },
    { 8.29434990883e-02, 2.03291710932e-04, 6.81538805366e-02, 9.70965325832e-01, },
    { 8.67543518543e-02, 2.10733342101e-04, 1.38366967440e-01, 2.55089461803e-01, }
    }, {
    { 5.96254616976e-02, 8.62886518007e-05, 3.61776389182e-02, 4.79704320431e-01, },
    { 6.12571723759e-02, 7.23929333617e-05, 3.93057651818e-02, 9.29222583771e-01, },
    { 6.58465921879e-02, 1.03639606095e-04, 6.07583411038e-02, 9.90289509296e-01, } }
  };
  // clang-format on

  float mX;
  float mAlpha;
  float mP[5];
  float mC[15];
  int mNDF = -5;
  float mCosAlpha;
  float mSinAlpha;

  // propagation parameters
  float mBz;
  MaterialCorrection mMaterial;

  PhysicalTrackModel mTrk;
#endif
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
