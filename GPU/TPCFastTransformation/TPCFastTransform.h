// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCFastTransform.h
/// \brief Definition of TPCFastTransform class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORM_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORM_H

#include "FlatObject.h"
#include "TPCFastTransformGeo.h"
#include "TPCFastSpaceChargeCorrection.h"
#include "GPUCommonMath.h"

#if !defined(GPUCA_GPUCODE)
#include <string>
#endif // !GPUCA_GPUCODE

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The TPCFastTransform class represents transformation of raw TPC coordinates to XYZ
///
/// (TPC Row number, Pad, Drift Time) ->  (X,Y,Z)
///
/// The following coordinate systems are used:
///
/// 1. raw coordinate system: TPC row number [int], readout pad number [float], drift time [float]
///
/// 2. drift volume coordinate system (x,u,v)[cm]. These are cartesian coordinates:
///    x = local x,
///    u = along the local y axis but towards to the pad increase direction,
///    v = along the global z axis but towards the drift length increase derection.
///
///    u and v are mirrored for A/C sides of the TPC
///
/// 3. local coordinate system: x,y,z, where global x,y are rotated such that x goes through the middle of the TPC sector
///
/// 4. global coordinate system: x,y,z in ALICE coordinate system
///
///
/// The transformation is pefformed as the following:
///
/// First, the class transforms input raw coordinates to the drift volume coordinates applying the drift velocity calibration.
/// Then it aplies TPCCorrectionIRS to the drift coordinates.
/// At the end it transforms the drift coordinates to the output local coordinates.
///
/// The class is flat C structure. No virtual methods, no ROOT types are used.

class TPCFastTransform : public FlatObject
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastTransform();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject() instead
  TPCFastTransform(const TPCFastTransform&) CON_DELETE;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject() instead
  TPCFastTransform& operator=(const TPCFastTransform&) CON_DELETE;

  /// Destructor
  ~TPCFastTransform() CON_DEFAULT;

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Memory alignment

  /// Gives minimal alignment in bytes required for the class object
  static constexpr size_t getClassAlignmentBytes() { return TPCFastSpaceChargeCorrection::getClassAlignmentBytes(); }

  /// Gives minimal alignment in bytes required for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() { return TPCFastSpaceChargeCorrection::getBufferAlignmentBytes(); }

  /// Construction interface

  void cloneFromObject(const TPCFastTransform& obj, char* newFlatBufferPtr);

  /// Making the data buffer external

  using FlatObject::releaseInternalBuffer;
  void moveBufferTo(char* newBufferPtr);

  /// Moving the class with its external buffer to another location

  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

  /// _______________  Construction interface  ________________________

  /// Starts the initialization procedure, reserves temporary memory
  void startConstruction(const TPCFastSpaceChargeCorrection& correction);

  /// Sets all drift calibration parameters and the time stamp
  ///
  /// It must be called once during construction,
  /// but also may be called afterwards to reset these parameters.
  void setCalibration(long int timeStamp, float t0, float vDrift, float vDriftCorrY, float lDriftCorr, float tofCorr, float primVtxZ);

  /// Sets the time stamp of the current calibaration
  void setTimeStamp(long int v) { mTimeStamp = v; }

  /// Gives a reference for external initialization of TPC corrections
  GPUd() const TPCFastSpaceChargeCorrection& getCorrection() const { return mCorrection; }

  /// Gives a reference for external initialization of TPC corrections
  TPCFastSpaceChargeCorrection& getCorrection() { return mCorrection; }

  /// Finishes initialization: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a slice
  /// taking calibration + alignment into account.
  ///
  GPUd() void Transform(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime = 0) const;

  /// Transformation in the time frame
  GPUd() void TransformInTimeFrame(int slice, int row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const;

  /// Inverse transformation
  GPUd() void InverseTransformInTimeFrame(int slice, int row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const;

  /// Inverse transformation: Transformed Y and Z -> transformed X
  GPUd() void InverseTransformYZtoX(int slice, int row, float y, float z, float& x) const;

  /// Inverse transformation: Transformed Y and Z -> Y and Z, transformed w/o space charge correction
  GPUd() void InverseTransformYZtoNominalYZ(int slice, int row, float y, float z, float& ny, float& nz) const;

  /// Ideal transformation with Vdrift only - without calibration
  GPUd() void TransformIdeal(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime) const;

  GPUd() void convPadTimeToUV(int slice, int row, float pad, float time, float& u, float& v, float vertexTime) const;
  GPUd() void convPadTimeToUVinTimeFrame(int slice, int row, float pad, float time, float& u, float& v, float maxTimeBin) const;

  GPUd() void convUVtoPadTime(int slice, int row, float u, float v, float& pad, float& time, float vertexTime) const;
  GPUd() void convUVtoPadTimeInTimeFrame(int slice, int row, float u, float v, float& pad, float& time, float maxTimeBin) const;
  GPUd() void convVtoTime(float v, float& time, float vertexTime) const;

  GPUd() float convTimeToZinTimeFrame(int slice, float time, float maxTimeBin) const;
  GPUd() float convZtoTimeInTimeFrame(int slice, float z, float maxTimeBin) const;
  GPUd() float convDeltaTimeToDeltaZinTimeFrame(int slice, float deltaTime) const;
  GPUd() float convDeltaZtoDeltaTimeInTimeFrame(int slice, float deltaZ) const;
  GPUd() float convZOffsetToVertexTime(int slice, float zOffset, float maxTimeBin) const;
  GPUd() float convVertexTimeToZOffset(int slice, float vertexTime, float maxTimeBin) const;

  GPUd() void getTOFcorrection(int slice, int row, float x, float y, float z, float& dz) const;

  void setApplyCorrectionOn() { mApplyCorrection = 1; }
  void setApplyCorrectionOff() { mApplyCorrection = 0; }
  bool isCorrectionApplied() { return mApplyCorrection; }

  /// _______________  Utilities  _______________________________________________

  /// TPC geometry information
  GPUd() const TPCFastTransformGeo& getGeometry() const { return mCorrection.getGeometry(); }

  /// Gives the time stamp of the current calibaration parameters
  GPUd() long int getTimeStamp() const { return mTimeStamp; }

  /// Return mVDrift in cm / time bin
  GPUd() float getVDrift() const { return mVdrift; }

  /// Return T0 in time bin units
  GPUd() float getT0() const { return mT0; }

  /// Return VdriftCorrY in time_bin / cn
  GPUd() float getVdriftCorrY() const { return mVdriftCorrY; }

  /// Return LdriftCorr offset in cm
  GPUd() float getLdriftCorr() const { return mLdriftCorr; }

  /// Return TOF correction (vdrift / C)
  GPUd() float getTOFCorr() const { return mLdriftCorr; }

  /// maximal possible drift timre of the active area
  GPUd() float getMaxDriftTime(int slice, int row, float pad) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int slice, int row) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int slice) const;

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

  int writeToFile(std::string outFName = "", std::string name = "");

  static TPCFastTransform* loadFromFile(std::string inpFName = "", std::string name = "");

#endif // !GPUCA_GPUCODE

  /// Print method
  void print() const;

 private:
  /// Enumeration of possible initialization states
  enum ConstructionExtraState : unsigned int {
    CalibrationIsSet = 0x4 ///< the drift calibration is set
  };

  /// _______________  Utilities  _______________________________________________

  /// _______________  Data members  _______________________________________________

  /// _______________  Calibration data. See Transform() method  ________________________________

  long int mTimeStamp; ///< time stamp of the current calibration

  /// Correction of (x,u,v) with irregular splines.
  ///
  /// After the initialization, mCorrection.getFlatBufferPtr()
  /// is pointed to the corresponding part of this->mFlatBufferPtr
  ///
  TPCFastSpaceChargeCorrection mCorrection;

  bool mApplyCorrection; // flag for applying correction

  /// _____ Parameters for drift length calculation ____
  ///
  /// t = (float) time bin, y = global y
  ///
  /// L(t,y) = (t-mT0)*(mVdrift + mVdriftCorrY*y ) + mLdriftCorr  ____
  ///
  float mT0;          ///< T0 in [time bin]
  float mVdrift;      ///< VDrift in  [cm/time bin]
  float mVdriftCorrY; ///< VDrift correction for global Y[cm] in [1/time bin]
  float mLdriftCorr;  ///< drift length correction in [cm]

  /// A coefficient for Time-Of-Flight correction: drift length -= EstimatedDistanceToVtx[cm]*mTOFcorr
  ///
  /// Since this correction requires a knowledge of the spatial position, it is appied after mCorrection,
  /// not on the drift length but directly on V coordinate.
  ///
  /// mTOFcorr == mVdrift/(speed of light)
  ///
  float mTOFcorr;

  float mPrimVtxZ; ///< Z of the primary vertex, needed for the Time-Of-Flight correction
#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(TPCFastTransform, 1);
#endif
};

// =======================================================================
//              Inline implementations of some methods
// =======================================================================

GPUdi() void TPCFastTransform::convPadTimeToUV(int slice, int row, float pad, float time, float& u, float& v, float vertexTime) const
{
  bool sideC = (slice >= getGeometry().getNumberOfSlicesA());

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  const TPCFastTransformGeo::SliceInfo& sliceInfo = getGeometry().getSliceInfo(slice);

  float x = rowInfo.x;
  u = (pad - 0.5 * rowInfo.maxPad) * rowInfo.padWidth;

  float y = sideC ? -u : u; // pads are mirrorred on C-side
  float yLab = y * sliceInfo.cosAlpha + x * sliceInfo.sinAlpha;

  v = (time - mT0 - vertexTime) * (mVdrift + mVdriftCorrY * yLab) + mLdriftCorr; // drift length cm
}

GPUdi() void TPCFastTransform::convPadTimeToUVinTimeFrame(int slice, int row, float pad, float time, float& u, float& v, float maxTimeBin) const
{
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  u = (pad - 0.5 * rowInfo.maxPad) * rowInfo.padWidth;
  v = (time - mT0 - maxTimeBin) * mVdrift + mLdriftCorr; // drift length cm
  if (slice < getGeometry().getNumberOfSlicesA()) {
    v += getGeometry().getTPCzLengthA();
  } else {
    v += getGeometry().getTPCzLengthC();
  }
}

GPUdi() float TPCFastTransform::convZOffsetToVertexTime(int slice, float zOffset, float maxTimeBin) const
{
  if (slice < getGeometry().getNumberOfSlicesA()) {
    return maxTimeBin - (getGeometry().getTPCzLengthA() + zOffset) / mVdrift;
  } else {
    return maxTimeBin - (getGeometry().getTPCzLengthC() - zOffset) / mVdrift;
  }
}

GPUdi() float TPCFastTransform::convVertexTimeToZOffset(int slice, float vertexTime, float maxTimeBin) const
{
  if (slice < getGeometry().getNumberOfSlicesA()) {
    return (maxTimeBin - vertexTime) * mVdrift - getGeometry().getTPCzLengthA();
  } else {
    return -((maxTimeBin - vertexTime) * mVdrift - getGeometry().getTPCzLengthC());
  }
}

GPUdi() void TPCFastTransform::convUVtoPadTime(int slice, int row, float u, float v, float& pad, float& time, float vertexTime) const
{
  bool sideC = (slice >= getGeometry().getNumberOfSlicesA());

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  const TPCFastTransformGeo::SliceInfo& sliceInfo = getGeometry().getSliceInfo(slice);

  pad = u / rowInfo.padWidth + 0.5 * rowInfo.maxPad;

  float x = rowInfo.x;
  float y = sideC ? -u : u; // pads are mirrorred on C-side
  float yLab = y * sliceInfo.cosAlpha + x * sliceInfo.sinAlpha;
  time = mT0 + vertexTime + (v - mLdriftCorr) / (mVdrift + mVdriftCorrY * yLab);
}

GPUdi() void TPCFastTransform::convVtoTime(float v, float& time, float vertexTime) const
{
  float yLab = 0.f;
  time = mT0 + vertexTime + (v - mLdriftCorr) / (mVdrift + mVdriftCorrY * yLab);
}

GPUdi() void TPCFastTransform::convUVtoPadTimeInTimeFrame(int slice, int row, float u, float v, float& pad, float& time, float maxTimeBin) const
{
  if (slice < getGeometry().getNumberOfSlicesA()) {
    v -= getGeometry().getTPCzLengthA();
  } else {
    v -= getGeometry().getTPCzLengthC();
  }
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  pad = u / rowInfo.padWidth + 0.5 * rowInfo.maxPad;
  time = mT0 + maxTimeBin + (v - mLdriftCorr) / mVdrift;
}

GPUdi() void TPCFastTransform::getTOFcorrection(int slice, int /*row*/, float x, float y, float z, float& dz) const
{
  // calculate time of flight correction for  z coordinate

  bool sideC = (slice >= getGeometry().getNumberOfSlicesA());
  float distZ = z - mPrimVtxZ;
  float dv = -GPUCommonMath::Sqrt(x * x + y * y + distZ * distZ) * mTOFcorr;
  dz = sideC ? dv : -dv;
}

GPUdi() void TPCFastTransform::Transform(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a slice
  /// taking calibration + alignment into account.
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);

  // const SliceInfo &sliceInfo = getSliceInfo( slice );
  // bool sideC = ( slice >= NumberOfSlices / 2 );

  x = rowInfo.x;
  float u = 0, v = 0;
  convPadTimeToUV(slice, row, pad, time, u, v, vertexTime);

  if (mApplyCorrection) {
    float dx, du, dv;
    mCorrection.getCorrection(slice, row, u, v, dx, du, dv);
    x += dx;
    u += du;
    v += dv;
  }

  getGeometry().convUVtoLocal(slice, u, v, y, z);

  float dzTOF = 0;
  getTOFcorrection(slice, row, x, y, z, dzTOF);
  z += dzTOF;
}

GPUdi() void TPCFastTransform::TransformInTimeFrame(int slice, int row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Corrections and Time-Of-Flight correction are not alpplied.
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  x = rowInfo.x;
  float u = 0, v = 0;
  convPadTimeToUVinTimeFrame(slice, row, pad, time, u, v, maxTimeBin);
  getGeometry().convUVtoLocal(slice, u, v, y, z);
}

GPUdi() void TPCFastTransform::InverseTransformInTimeFrame(int slice, int row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const
{
  /// Inverse transformation to TransformInTimeFrame
  float u = 0, v = 0;
  getGeometry().convLocalToUV(slice, y, z, u, v);
  convUVtoPadTimeInTimeFrame(slice, row, u, v, pad, time, maxTimeBin);
}

GPUdi() void TPCFastTransform::TransformIdeal(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a slice
  /// Ideal transformation: only Vdrift from DCS.
  /// No space charge corrections, no time of flight correction
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);

  x = rowInfo.x;
  float u = (pad - 0.5 * rowInfo.maxPad) * rowInfo.padWidth;
  float v = (time - mT0 - vertexTime) * mVdrift; // drift length cm

  getGeometry().convUVtoLocal(slice, u, v, y, z);
}

GPUdi() float TPCFastTransform::convTimeToZinTimeFrame(int slice, float time, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Corrections and Time-Of-Flight correction are not alpplied.
  /// Only Z coordinate.
  ///

  float v = (time - mT0 - maxTimeBin) * mVdrift + mLdriftCorr; // drift length cm
  float z = getGeometry().getTPCalignmentZ();                  // global TPC alignment
  if (slice < getGeometry().getNumberOfSlicesA()) {
    z -= v;
  } else {
    z += v;
  }
  return z;
}

GPUdi() float TPCFastTransform::convZtoTimeInTimeFrame(int slice, float z, float maxTimeBin) const
{
  /// Inverse transformation of convTimeToZinTimeFrame()
  float v;
  if (slice < getGeometry().getNumberOfSlicesA()) {
    v = getGeometry().getTPCalignmentZ() - z;
  } else {
    v = z - getGeometry().getTPCalignmentZ();
  }
  return mT0 + maxTimeBin + (v - mLdriftCorr) / mVdrift;
}

GPUdi() float TPCFastTransform::convDeltaTimeToDeltaZinTimeFrame(int slice, float deltaTime) const
{
  float deltaZ = deltaTime * mVdrift;
  return slice < getGeometry().getNumberOfSlicesA() ? -deltaZ : deltaZ;
}

GPUdi() float TPCFastTransform::convDeltaZtoDeltaTimeInTimeFrame(int slice, float deltaZ) const
{
  float deltaT = deltaZ / mVdrift;
  return slice < getGeometry().getNumberOfSlicesA() ? -deltaT : deltaT;
}

/*
GPUdi() float TPCFastTransform::getLastCalibratedTimeBin(int slice) const
{
  /// Return a value of the last timebin where correction map is valid
  float u, v, pad, time;
  getGeometry().convScaledUVtoUV(slice, 0, 0.f, 1.f, u, v);
  convUVtoPadTime(slice, 0, u, v, pad, time, 0);
  return time;
}
*/

GPUdi() float TPCFastTransform::getMaxDriftTime(int slice, int row, float pad) const
{
  /// maximal possible drift time of the active area
  float maxL = mCorrection.getMaxDriftLength(slice, row, pad);

  bool sideC = (slice >= getGeometry().getNumberOfSlicesA());
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  const TPCFastTransformGeo::SliceInfo& sliceInfo = getGeometry().getSliceInfo(slice);

  float x = rowInfo.x;
  float u = (pad - 0.5 * rowInfo.maxPad) * rowInfo.padWidth;

  float y = sideC ? -u : u; // pads are mirrorred on C-side
  float yLab = y * sliceInfo.cosAlpha + x * sliceInfo.sinAlpha;
  return mT0 + (maxL - mLdriftCorr) / (mVdrift + mVdriftCorrY * yLab);
}

GPUdi() float TPCFastTransform::getMaxDriftTime(int slice, int row) const
{
  /// maximal possible drift time of the active area
  float maxL = mCorrection.getMaxDriftLength(slice, row);
  float maxTime = 0.f;
  convVtoTime(maxL, maxTime, 0.f);
  return maxTime;
}

GPUdi() float TPCFastTransform::getMaxDriftTime(int slice) const
{
  /// maximal possible drift time of the active area
  float maxL = mCorrection.getMaxDriftLength(slice);
  float maxTime = 0.f;
  convVtoTime(maxL, maxTime, 0.f);
  return maxTime;
}

GPUdi() void TPCFastTransform::InverseTransformYZtoX(int slice, int row, float y, float z, float& x) const
{
  /// Transformation y,z -> x
  float u = 0, v = 0;
  getGeometry().convLocalToUV(slice, y, z, u, v);
  mCorrection.getCorrectionInvCorrectedX(slice, row, u, v, x);
}

GPUdi() void TPCFastTransform::InverseTransformYZtoNominalYZ(int slice, int row, float y, float z, float& ny, float& nz) const
{
  /// Transformation y,z -> x
  float u = 0, v = 0;
  getGeometry().convLocalToUV(slice, y, z, u, v);
  mCorrection.getCorrectionInvUV(slice, row, u, v, u, v);
  getGeometry().convUVtoLocal(slice, u, v, ny, nz);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
