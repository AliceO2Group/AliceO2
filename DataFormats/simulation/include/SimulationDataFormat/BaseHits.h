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
#include "MathUtils/Cartesian3D.h"

namespace o2
{

// Mother class of all hit classes for AliceO2
// just defines what is absolutely necessary to have
// as common interface
// at the moment ony GetTrackID() used by Stack.h
// eventually we could add some interfaces to retrieve
// the coordinates as floats or something
class BaseHit
{
 public:
  BaseHit() = default;
  BaseHit(int id) : mTrackID{id} {}
  int GetTrackID() const { return mTrackID; }
  void SetTrackID(int id) { mTrackID = id; }

 private:
  int mTrackID = 0; // track_id
  ClassDefNV(BaseHit, 1);
};

// a set of configurable classes to define basic hit types
// these are meant to be an alternative to FairMCPoint
// which always includes the momentum and is purely based on double values

// Generic class to keep position, time and hit value
// T is basic type for position,
// E is basic type for time,
// V is basic type for hit value.
template <typename T, typename E, typename V = float>
class BasicXYZVHit : public BaseHit
{
  Point3D<T> mPos;   // cartesian position of Hit
  E mTime;           // time of flight
  V mHitValue;       // hit value
  short mDetectorID; // the detector/sensor id

 public:
  BasicXYZVHit() = default; // for ROOT IO

  // constructor
  BasicXYZVHit(T x, T y, T z, E time, V val, int trackid, short did)
    : mPos(x, y, z), mTime(time), mHitValue(val), BaseHit(trackid), mDetectorID(did)
  {
  }

  // getting the cartesian coordinates
  T GetX() const { return mPos.X(); }
  T GetY() const { return mPos.Y(); }
  T GetZ() const { return mPos.Z(); }
  Point3D<T> GetPos() const { return mPos; }
  // getting hit value
  V GetHitValue() const { return mHitValue; }
  // getting the time
  E GetTime() const { return mTime; }
  // get detector + track information
  short GetDetectorID() const { return mDetectorID; }

  // modifiers
  void SetTime(E time) { mTime = time; }
  void SetHitValue(V val) { mHitValue = val; }
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
  void SetPos(Point3D<T> const& p) { mPos = p; }

  ClassDefNV(BasicXYZVHit, 1);
};

// Class for a hit containing energy loss as hit value
// T is basic type for position,
// E is basic type for time (float as default),
// V is basic type for hit value (float as default).
template <typename T, typename E = float, typename V = float>
class BasicXYZEHit : public BasicXYZVHit<T, E, V>
{
 public:
  using BasicXYZVHit<T, E, V>::BasicXYZVHit;

  V GetEnergyLoss() const { return BasicXYZVHit<T, E, V>::GetHitValue(); }
  void SetEnergyLoss(V val) { BasicXYZVHit<T, E, V>::SetHitValue(val); }

  ClassDefNV(BasicXYZEHit, 1);
};

// Class for a hit containing charge as hit value
// T is basic type for position,
// E is basic type for time (float as default),
// V is basic type for hit value (int as default).
template <typename T, typename E = float, typename V = int>
class BasicXYZQHit : public BasicXYZVHit<T, E, V>
{
 public:
  using BasicXYZVHit<T, E, V>::BasicXYZVHit;

  V GetCharge() const { return BasicXYZVHit<T, E, V>::GetHitValue(); }
  void SetCharge(V val) { BasicXYZVHit<T, E, V>::SetHitValue(val); }

  ClassDefNV(BasicXYZQHit, 1);
};

} // namespace o2
#endif
