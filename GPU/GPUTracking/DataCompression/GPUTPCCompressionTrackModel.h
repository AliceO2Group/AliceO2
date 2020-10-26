// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCompressionTrackModel.h
/// \author David Rohr

#ifndef GPUTPCCOMPRESSIONTRACKMODEL_H
#define GPUTPCCOMPRESSIONTRACKMODEL_H

// For debugging purposes, we provide means to use other track models
#define GPUCA_COMPRESSION_TRACK_MODEL_MERGER
// #define GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER

#include "GPUDef.h"

#ifdef GPUCA_COMPRESSION_TRACK_MODEL_MERGER
#include "GPUTPCGMPropagator.h"
#include "GPUTPCGMTrackParam.h"

#elif defined(GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER)
#include "GPUTPCTrackParam.h"

#else // Default internal track model for compression
#include "GPUTPCGMPolynomialField.h"
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
  GPUd() void getBxByBz(float cosAlpha, float sinAlpha, float x, float y, float z, float b[3]) const;
  GPUd() float getBz(float x, float y, float z) const;
  GPUd() void updatePhysicalTrackValues(PhysicalTrackModel& trk);
  GPUd() void changeDirection();
  GPUd() int rotateToAlpha(float newAlpha);
  GPUd() int propagateToXBxByBz(PhysicalTrackModel& t, float x, float Bx, float By, float Bz, float& dLp);
  GPUd() int propagateToXBzLightNoUpdate(PhysicalTrackModel& t, float x, float Bz, float& dLp);
  GPUd() bool setDirectionAlongX(PhysicalTrackModel& t);
  GPUd() int followLinearization(const PhysicalTrackModel& t0e, float Bz, float dLp);
  GPUd() void calculateMaterialCorrection();
  GPUd() float approximateBetheBloch(float beta2);
  GPUd() void resetCovariance();

#endif

 protected:
  const GPUParam* mParam;

#ifdef GPUCA_COMPRESSION_TRACK_MODEL_MERGER
  GPUTPCGMPropagator mProp;
  GPUTPCGMTrackParam mTrk;

#elif defined(GPUCA_COMPRESSION_TRACK_MODEL_SLICETRACKER)
  GPUTPCTrackParam mTrk;
  float mAlpha;

#else // Default internal track model for compression

  struct MaterialCorrection {
    GPUhd() MaterialCorrection() : radLen(28811.7f), rho(1.025e-3f), radLenInv(1.f / radLen), DLMax(0.f), EP2(0.f), sigmadE2(0.f), k22(0.f), k33(0.f), k43(0.f), k44(0.f) {}

    float radLen;                                              // [cm]
    float rho;                                                 // [g/cm^3]
    float radLenInv, DLMax, EP2, sigmadE2, k22, k33, k43, k44; // precalculated values for MS and EnergyLoss correction
  };

  float mX;
  float mAlpha;
  float mP[5];
  float mC[15];
  int mNDF = -5;
  float mChi2 = 0.f;
  float mCosAlpha;
  float mSinAlpha;

  // propagation parameters
  const GPUTPCGMPolynomialField* mField = nullptr;
  MaterialCorrection mMaterial;

  PhysicalTrackModel mTrk;
#endif
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
