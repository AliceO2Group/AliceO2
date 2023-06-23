// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_COORDINATE_TRANSFORMER_H_
#define ALICEO2_TRD_COORDINATE_TRANSFORMER_H_

///
/// \file   CoordinateTransformer.h
/// \author Thomas Dietel, tom@dietel.net
///

#include "DataFormatsTRD/Hit.h"

#include <array>

namespace o2::trd
{

class Geometry;


// /// A position in spatial (x,y,z) and raw/digit coordinates (det,row,col,tb).
// The class was used in a previous implementation, and still needs to be adopted for inclusion into O2.
// class ChamberSpacePoint
// {
//  public:
//   ChamberSpacePoint(int det = -999) : mDetector(det){};
//   // ChamberSpacePoint(o2::track::TrackParCov& t);

//   /// check if the space point has been initialized
//   bool isValid() const { return mDetector >= 0; }

//   /// spatial x coordinate of space point
//   float getX() const { return mX; }

//   /// spatial y coordinate of space point
//   float getY() const { return mY; }

//   /// spatial z coordinate of space point
//   float getZ() const { return mZ; }

//   /// detector number corresponding to space point
//   int getDetector() const { return mDetector; }

//   /// pad row within detector of space point 
//   int getPadRow() const { return mPadrow; }

//   /// pad position (a.k.a. column) within pad row
//   float getPadCol() const { return mPadcol; }

//   /// time coordinate in drift direction
//   float getTimeBin() const { return mTimebin; }

//   /// calculate MCM corresponding to pad row/column
//   // int getMCM() const { return o2::trd::HelperMethods::getMCMfromPad(mPadrow, mPadcol); }

//   /// calculate readout board corresponding to pad row/column
//   // int getROB() const { return o2::trd::HelperMethods::getROBfromPad(mPadrow, mPadcol); }

//  protected:
//   float mX, mY, mZ;
//   int mDetector;
//   int mPadrow;
//   float mPadcol, mTimebin;

//   static constexpr float xscale = 1.0 / (o2::trd::Geometry::cheight() + o2::trd::Geometry::cspace());
//   static constexpr float xoffset = o2::trd::Geometry::getTime0(0);
//   static constexpr float alphascale = 1.0 / o2::trd::Geometry::getAlpha();
// };

// std::ostream& operator<<(std::ostream& os, const ChamberSpacePoint& p);

/// CoordinateTransformer: translate between local spatial and pad/timebin coordinates
///
/// The intention of the CoordinateTransformer is to bundle calculations around coordinate transformations for the TRD and access to the calibration objects.
/// So far it translates local spatial (x/y/z) coordinates within a TRD chamber (detector) to a tuple of pad row, pad column and time bin.
/// In the future, it could translate e.g. track parameters to expected tracklet position and inclination.
/// At the time of writing, only constant drift velocity, Lorentz angle and T0 are considered, and must be set by hand.
class CoordinateTransformer
{
public:
  static CoordinateTransformer* instance()
  {
    static CoordinateTransformer mCTrans;
    return &mCTrans;
  }

  /// Translate local spatial (x/y/z) coordinates within a TRD chamber (detector) to a tuple of pad row, column and time bin.
  ///
  /// Local x,y,z coordinates follow the convention for MC hist and are assumed to be corrrected for alignment.
  /// The x-coordinate points in time direction, y in pad and z in row direction.
  /// The result is an array of three floating point numbers for row, column and timebin.
  /// Time and column can directly be compared with digit or tracklet data.
  /// The pad row is returned as a floating point number that indicates also the position within the padrow.
  /// The fractional part of the pad row is not available for digits and tracklets, and only 
  std::array<float, 3> Local2RCT(int det, float x, float y, float z);

  /// Wrapper to conveniently calculate the row/column/time coordinate of a MC hit.
  std::array<float, 3> Local2RCT(o2::trd::Hit& hit) 
  { return Local2RCT(hit.GetDetectorID(), hit.getLocalT(), hit.getLocalC(), hit.getLocalR()); }

  /// Legacy, less accurate method to convert local spatial to row/column/time coordinate.
  /// This method is only included for comparision, and should be removed in the future.
  std::array<float, 3> OrigLocal2RCT(int det, float x, float y, float z);

  /// Legacy, less accurate method to convert local spatial to row/column/time coordinate.
  /// This method is only included for comparision, and should be removed in the future.
  std::array<float, 3> OrigLocal2RCT(o2::trd::Hit& hit) 
  { return OrigLocal2RCT(hit.GetDetectorID(), hit.getLocalT(), hit.getLocalC(), hit.getLocalR()); }

  float GetVdrift() { return mVdrift; }
  void SetVdrift(float x) { mVdrift = x; }

  float GetT0() { return mT0; }
  void SetT0(float x) { mT0 = x; }

  float GetExB() { return mExB; }
  void SetExB(float x) { mExB = x; }

protected:
  o2::trd::Geometry* mGeo;
  float mVdrift{1.5625}; ///< drift velocity in cm/us
  float mT0{4.0}; ///< time offset of start of drift region
  float mExB{0.140}; ///< tan(Lorentz angle): tan(8 deg) ~ 0.140

private:
  CoordinateTransformer();

};


} // namespace o2::trd

#endif // ALICEO2_TRD_RAWDISPLAY_H_
