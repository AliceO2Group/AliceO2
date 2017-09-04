// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_BASE_HIT_H
#define ALICEO2_BASE_HIT_H
#include "FairMultiLinkedData_Interface.h"
#include "MathUtils/Cartesian3D.h"

namespace o2
{

// Mother class of all hit classes for AliceO2
// just defines what is absolutely necessary to have
// as common interface
// at the moment ony GetTrackID() used by Stack.h
// eventually we could add some interfaces to retrieve
// the coordinates as floats or something
class BaseHit : public FairMultiLinkedData_Interface
{
 public:
  BaseHit() = default;
  BaseHit(int id) : mTrackID{id} {}
  int GetTrackID() const { return mTrackID; }
  void SetTrackID(int id) { mTrackID = id; }

 private:
  int mTrackID; // track_id
  ClassDefOverride(BaseHit, 1);
};

// a set of configurable classes to define basic hit types
// these are meant to be an alternative to FairMCPoint
// which always includes the momentum and is purely based on double values

// T is basic type for position, E is basic type for time and energy loss
template <typename T, typename E=float>
class BasicXYZEHit : public BaseHit
{
  Point3D<T> mPos; // cartesian position of Hit
  E mTime;         // time of flight
  E mELoss;        // energy loss
  short mDetectorID; // the detector/sensor id

 public:
  BasicXYZEHit() = default; // for ROOT IO

  // constructor
  BasicXYZEHit(T x, T y, T z, E time, E e, int trackid, short did)
    :  mPos(x, y, z), mTime(time), mELoss(e), BaseHit(trackid), mDetectorID(did)
  {
  }

  // getting the cartesian coordinates
  T GetX() const { return mPos.X(); }
  T GetY() const { return mPos.Y(); }
  T GetZ() const { return mPos.Z(); }
  Point3D<T> GetPos() const { return mPos; }
  // getting energy loss
  E GetEnergyLoss() const { return mELoss; }
  // getting the time
  E GetTime() const { return mTime; }
  // get detector + track information
  short GetDetectorID() const { return mDetectorID; }

  // modifiers
  void SetTime(E time) { mTime = time; }
  void SetEnergyLoss(E eLoss) { mELoss = eLoss; }
  void SetDetectorID(short detID) { mDetectorID = detID; }
  void SetX(T x) { mPos.SetX(x); }
  void SetY(T y) { mPos.SetY(y); }
  void SetZ(T z) { mPos.SetZ(z); }
  void SetXYZ(T x, T y, T z)
  {
    SetX(x);
    SetY(y);
    SetZ(z);
  }
  void SetPos(Point3D<T> const &p) { mPos = p; }

  ClassDefOverride(BasicXYZEHit, 1);
};

} // end namespace AliceO2
#endif
