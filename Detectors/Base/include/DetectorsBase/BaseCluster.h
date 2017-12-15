// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_BASE_BASECLUSTER_H
#define ALICEO2_BASE_BASECLUSTER_H

#include <TObject.h>
#include <bitset>
#include <iomanip>
#include <ios>
#include <iostream>
#include "MathUtils/Cartesian3D.h"
#include "DetectorsBase/DetMatrixCache.h"
namespace o2
{
namespace Base 
{
  
// Basic cluster class with X,Y,Z position detector ID information + user fields
// The position is ALWAYS stored in tracking frame and is misaligned (in opposite
// to AliRoot). The errors are defined in *ideal* tracking frame
// DetectorID should correspond to continuous (no jumps between detector layers
// planes etc.) internal sensor ID within detector
// Detector specific clusters should be composed by including it as data member
template <typename T>
class BaseCluster : public TObject // temprarily derive from TObject
{
 private:
  
  Point3D<T> mPos;        // cartesian position
  T mSigmaY2;             // error in Y direction (usually rphi)
  T mSigmaZ2;             // error in Z direction (usually Z)
  T mSigmaYZ;             // non-diagonal term of error matrix
  std::uint16_t mSensorID=0; // the sensor id
  std::int8_t mCount = 0; // user field reserved for counting
  std::uint8_t mBits = 0; // user field reserved for bit flags
  enum masks_t : std::int32_t { kUserBitsMask = 0xff };

 public:
  BaseCluster() = default;

  // constructor
  BaseCluster(std::uint16_t sensid, const Point3D<T>& xyz) : mPos(xyz), mSensorID(sensid) {}
  BaseCluster(std::uint16_t sensid, T x, T y, T z) : mPos(x, y, z), mSensorID(sensid) {}
  BaseCluster(std::uint16_t sensid, const Point3D<T>& xyz, T sy2, T sz2, T syz)
    : mPos(xyz), mSigmaY2(sy2), mSigmaZ2(sz2), mSigmaYZ(syz), mSensorID(sensid)
  {
  }
  BaseCluster(std::int16_t sensid, T x, T y, T z, T sy2, T sz2, T syz)
    : mPos(x, y, z), mSigmaY2(sy2), mSigmaZ2(sz2), mSigmaYZ(syz), mSensorID(sensid)
  {
  }

  // getting the cartesian coordinates and errors
  T getX() const { return mPos.X(); }
  T getY() const { return mPos.Y(); }
  T getZ() const { return mPos.Z(); }
  T getSigmaY2() const { return mSigmaY2; }
  T getSigmaZ2() const { return mSigmaZ2; }
  T getSigmaYZ() const { return mSigmaYZ; }
  Point3D<T>  getXYZ() const { return mPos; }
  Point3D<T>& getXYZ() { return mPos; }

  // position in local frame, no check for matrices cache validity 
  Point3D<T> getXYZLoc(const DetMatrixCache& dm) const {
    return dm.getMatrixT2L(mSensorID)(mPos);
  }

  // position in global frame, no check for matrices cache validity 
  Point3D<T> getXYZGlo(const DetMatrixCache& dm) const {
    return dm.getMatrixT2G(mSensorID)(mPos);
  }

  // position in global frame obtained as simple rotation from tracking one:
  // much faster for barrel detectors than using full 3D matrix.
  // no check for matrices cache validity 
  Point3D<T> getXYZGloRot(const DetMatrixCache& dm) const {
    return dm.getMatrixT2GRot(mSensorID)(mPos);
  }
  
  // get sensor id
  std::int16_t getSensorID() const { return mSensorID; }
  // get count field
  std::int8_t getCount() const { return mCount; }
  // get bit field
  std::uint8_t getBits() const { return mBits; }
  bool isBitSet(int bit) const { return mBits & (0xff & (0x1 << bit)); }

  // cast to Point3D
  operator Point3D<T>&() { return mPos; }
  // modifiers

  // set sensor id
  void setSensorID(std::int16_t sid) { mSensorID = sid; }
  // set count field
  void setCount(std::int8_t c) { mCount = c; }
  // set bit field
  void setBits(std::uint8_t b) { mBits = b; }
  void setBit(int bit) { mBits |= kUserBitsMask & (0x1 << bit); }
  void resetBit(int bit) { mBits &= ~(kUserBitsMask & (0x1 << bit)); }

  // set position and errors
  void setX(T x) { mPos.SetX(x); }
  void setY(T y) { mPos.SetY(y); }
  void setZ(T z) { mPos.SetZ(z); }
  void setXYZ(T x, T y, T z)
  {
    setX(x);
    setY(y);
    setZ(z);
  }
  void setPos(const Point3D<T>& p) { mPos = p; }
  void setSigmaY2(T v) { mSigmaY2 = v; }
  void setSigmaZ2(T v) { mSigmaZ2 = v; }
  void setSigmaYZ(T v) { mSigmaYZ = v; }
  void setErrors(T sy2, T sz2, T syz)
  {
    setSigmaY2(sy2);
    setSigmaZ2(sz2);
    setSigmaYZ(syz);
  }

 protected:
  ~BaseCluster() override = default;
  
  //  ClassDefNV(BaseCluster, 1);
  ClassDefOverride(BaseCluster, 1); // temporarily
};

template <class T>
std::ostream& operator<<(std::ostream& os, const BaseCluster<T>& c)
{
  // stream itself
  os << "SId" << std::setw(5) << c.getSensorID() << " (" << std::showpos << std::scientific << c.getX() << ","
     << std::scientific << c.getY() << "," << std::scientific << c.getZ() << ") cnt:" << std::setw(4) << +c.getCount()
     << " bits:" << std::bitset<8>(c.getBits());
  return os;
}
}
} // end namespace AliceO2
#endif
