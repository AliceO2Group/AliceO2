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

/// \file GPUTPCGMPhysicalTrackModel.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCGMPHYSICALTRACKMODEL_H
#define GPUTPCGMPHYSICALTRACKMODEL_H

#include "GPUTPCGMTrackParam.h"

/**
 * @class GPUTPCGMPhysicalTrackModel
 *
 * GPUTPCGMPhysicalTrackModel class is a trajectory in physical parameterisation (X,Y,Z,Px,PY,Pz,Q)
 * without covariance matrix. Px>0 and Q is {-1,+1} (no uncharged tracks).
 *
 * It is used to linearise transport equations for GPUTPCGMTrackParam trajectory during (re)fit.
 *
 */

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCGMPhysicalTrackModel
{
 public:
  GPUdDefault() GPUTPCGMPhysicalTrackModel() CON_DEFAULT;
  GPUd() GPUTPCGMPhysicalTrackModel(const GPUTPCGMTrackParam& t);

  GPUd() void Set(const GPUTPCGMTrackParam& t);
  GPUd() void Set(float X, float Y, float Z, float Px, float Py, float Pz, float Q);

  GPUd() float& X()
  {
    return mX;
  }
  GPUd() float& Y()
  {
    return mY;
  }
  GPUd() float& Z()
  {
    return mZ;
  }
  GPUd() float& Px()
  {
    return mPx;
  }
  GPUd() float& Py()
  {
    return mPy;
  }
  GPUd() float& Pz()
  {
    return mPz;
  }
  GPUd() float& Q()
  {
    return mQ;
  }

  GPUd() float& SinPhi()
  {
    return mSinPhi;
  }
  GPUd() float& CosPhi()
  {
    return mCosPhi;
  }
  GPUd() float& SecPhi()
  {
    return mSecPhi;
  }
  GPUd() float& DzDs()
  {
    return mDzDs;
  }
  GPUd() float& DlDs()
  {
    return mDlDs;
  }
  GPUd() float& QPt()
  {
    return mQPt;
  }
  GPUd() float& P()
  {
    return mP;
  }
  GPUd() float& Pt()
  {
    return mPt;
  }

  GPUd() const float& SinPhi() const { return mSinPhi; }
  GPUd() const float& DzDs() const { return mDzDs; }

  GPUd() float GetX() const { return mX; }
  GPUd() float GetY() const { return mY; }
  GPUd() float GetZ() const { return mZ; }
  GPUd() float GetPx() const { return mPx; }
  GPUd() float GetPy() const { return mPy; }
  GPUd() float GetPz() const { return mPz; }
  GPUd() float GetQ() const { return mQ; }

  GPUd() float GetSinPhi() const { return mSinPhi; }
  GPUd() float GetCosPhi() const { return mCosPhi; }
  GPUd() float GetSecPhi() const { return mSecPhi; }
  GPUd() float GetDzDs() const { return mDzDs; }
  GPUd() float GetDlDs() const { return mDlDs; }
  GPUd() float GetQPt() const { return mQPt; }
  GPUd() float GetP() const { return mP; }
  GPUd() float GetPt() const { return mPt; }

  GPUd() int PropagateToXBzLightNoUpdate(float x, float Bz, float& dLp);
  GPUd() int PropagateToXBzLight(float x, float Bz, float& dLp);

  GPUd() int PropagateToXBxByBz(float x, float Bx, float By, float Bz, float& dLp);

  GPUd() int PropagateToLpBz(float Lp, float Bz);

  GPUd() bool SetDirectionAlongX();

  GPUd() void UpdateValues();

  GPUd() void Print() const;

  GPUd() float GetMirroredY(float Bz) const;

  GPUd() void Rotate(float alpha);
  GPUd() void RotateLight(float alpha);

 private:
  // physical parameters of the trajectory

  float mX = 0.f;    // X
  float mY = 0.f;    // Y
  float mZ = 0.f;    // Z
  float mPx = 1.e4f; // Px, >0
  float mPy = 0.f;   // Py
  float mPz = 0.f;   // Pz
  float mQ = 1.f;    // charge, +-1

  // some additional variables needed for GMTrackParam transport

  float mSinPhi = 0.f; // SinPhi = Py/Pt
  float mCosPhi = 1.f; // CosPhi = abs(Px)/Pt
  float mSecPhi = 1.f; // 1/cos(phi) = Pt/abs(Px)
  float mDzDs = 0.f;   // DzDs = Pz/Pt
  float mDlDs = 0.f;   // DlDs = P/Pt
  float mQPt = 0.f;    // QPt = q/Pt
  float mP = 1.e4f;    // momentum
  float mPt = 1.e4f;   // Pt momentum
};

GPUdi() GPUTPCGMPhysicalTrackModel::GPUTPCGMPhysicalTrackModel(const GPUTPCGMTrackParam& t) { Set(t); }

GPUdi() void GPUTPCGMPhysicalTrackModel::Set(const GPUTPCGMTrackParam& GPUrestrict() t)
{
  float pti = CAMath::Abs(t.GetQPt());
  if (pti < 1.e-4f) {
    pti = 1.e-4f; // set 10000 GeV momentum for straight track
  }
  mQ = (t.GetQPt() >= 0) ? 1.f : -1.f; // only charged tracks are considered
  mX = t.GetX();
  mY = t.GetY();
  mZ = t.GetZ();

  mPt = 1.f / pti;
  mSinPhi = t.GetSinPhi();
  if (mSinPhi > GPUCA_MAX_SIN_PHI) {
    mSinPhi = GPUCA_MAX_SIN_PHI;
  }
  if (mSinPhi < -GPUCA_MAX_SIN_PHI) {
    mSinPhi = -GPUCA_MAX_SIN_PHI;
  }
  mCosPhi = CAMath::Sqrt((1.f - mSinPhi) * (1.f + mSinPhi));
  mSecPhi = 1.f / mCosPhi;
  mDzDs = t.GetDzDs();
  mDlDs = CAMath::Sqrt(1.f + mDzDs * mDzDs);
  mP = mPt * mDlDs;

  mPy = mPt * mSinPhi;
  mPx = mPt * mCosPhi;
  mPz = mPt * mDzDs;
  mQPt = mQ * pti;
}

GPUdi() void GPUTPCGMPhysicalTrackModel::Set(float X, float Y, float Z, float Px, float Py, float Pz, float Q)
{
  mX = X;
  mY = Y;
  mZ = Z;
  mPx = Px;
  mPy = Py;
  mPz = Pz;
  mQ = (Q >= 0) ? 1 : -1;
  UpdateValues();
}

GPUdi() void GPUTPCGMPhysicalTrackModel::UpdateValues()
{
  float px = mPx;
  if (CAMath::Abs(px) < 1.e-4f) {
    px = copysign(1.e-4f, px);
  }

  mPt = CAMath::Sqrt(px * px + mPy * mPy);
  float pti = 1.f / mPt;
  mP = CAMath::Sqrt(px * px + mPy * mPy + mPz * mPz);
  mSinPhi = mPy * pti;
  mCosPhi = px * pti;
  mSecPhi = mPt / px;
  mDzDs = mPz * pti;
  mDlDs = mP * pti;
  mQPt = mQ * pti;
}

GPUdi() bool GPUTPCGMPhysicalTrackModel::SetDirectionAlongX()
{
  //
  // set direction of movenment collinear to X axis
  // return value is true when direction has been changed
  //
  if (mPx >= 0) {
    return 0;
  }

  mPx = -mPx;
  mPy = -mPy;
  mPz = -mPz;
  mQ = -mQ;
  UpdateValues();
  return 1;
}

GPUdi() float GPUTPCGMPhysicalTrackModel::GetMirroredY(float Bz) const
{
  // get Y of the point which has the same X, but located on the other side of trajectory
  if (CAMath::Abs(Bz) < 1.e-8f) {
    Bz = 1.e-8f;
  }
  return mY - 2.f * mQ * mPx / Bz;
}

GPUdi() void GPUTPCGMPhysicalTrackModel::RotateLight(float alpha)
{
  //* Rotate the coordinate system in XY on the angle alpha

  float cA = CAMath::Cos(alpha);
  float sA = CAMath::Sin(alpha);
  float x = mX, y = mY, px = mPx, py = mPy;
  mX = x * cA + y * sA;
  mY = -x * sA + y * cA;
  mPx = px * cA + py * sA;
  mPy = -px * sA + py * cA;
}

GPUdi() void GPUTPCGMPhysicalTrackModel::Rotate(float alpha)
{
  //* Rotate the coordinate system in XY on the angle alpha
  RotateLight(alpha);
  UpdateValues();
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
