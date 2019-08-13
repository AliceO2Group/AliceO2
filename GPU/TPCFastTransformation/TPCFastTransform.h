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
#include "TPCDistortionIRS.h"
#include "GPUCommonMath.h"

#if !defined(GPUCA_GPUCODE)
#include <string>
#endif // !GPUCA_GPUCODE

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
//#include "Rtypes.h"
#endif

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
  static constexpr size_t getClassAlignmentBytes() { return TPCDistortionIRS::getClassAlignmentBytes(); }

  /// Gives minimal alignment in bytes required for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() { return TPCDistortionIRS::getBufferAlignmentBytes(); }

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
  void startConstruction(const TPCDistortionIRS& distortion);

  /// Sets all drift calibration parameters and the time stamp
  ///
  /// It must be called once during construction,
  /// but also may be called afterwards to reset these parameters.
  void setCalibration(long int timeStamp, float t0, float vDrift, float vDriftCorrY, float lDriftCorr, float tofCorr, float primVtxZ);

  /// Sets the time stamp of the current calibaration
  void setTimeStamp(long int v) { mTimeStamp = v; }

  /// Gives a reference for external initialization of TPC distortions
  GPUd() const TPCDistortionIRS& getDistortion() const { return mDistortion; }

  /// Gives a reference for external initialization of TPC distortions
  TPCDistortionIRS& getDistortionNonConst() { return mDistortion; }

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
  GPUd() void InverseTransformInTimeFrame(int slice, int row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const;

  /// Ideal transformation with Vdrift only - without calibration
  GPUd() void TransformIdeal(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime) const;

  GPUd() void convPadTimeToUV(int slice, int row, float pad, float time, float& u, float& v, float vertexTime) const;
  GPUd() void convPadTimeToUVinTimeFrame(int slice, int row, float pad, float time, float& u, float& v, float maxTimeBin) const;

  GPUd() void convUVtoPadTime(int slice, int row, float u, float v, float& pad, float& time, float vertexTime) const;
  GPUd() void convUVtoPadTimeInTimeFrame(int slice, int row, float u, float v, float& pad, float& time, float maxTimeBin) const;

  GPUd() void getTOFcorrection(int slice, int row, float x, float y, float z, float& dz) const;

  void setApplyDistortionOn() { mApplyDistortion = 1; }
  void setApplyDistortionOff() { mApplyDistortion = 0; }
  bool isDistortionApplied() { return mApplyDistortion; }

  /// _______________  Utilities  _______________________________________________

  /// TPC geometry information
  GPUd() const TPCFastTransformGeo& getGeometry() const { return mDistortion.getGeometry(); }

  /// Gives the time stamp of the current calibaration parameters
  long int getTimeStamp() const { return mTimeStamp; }

  /// Return mVDrift in cm / time bin
  GPUd() float getVDrift() const { return mVdrift; }

#if !defined(GPUCA_GPUCODE)

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
  /// After the initialization, mDistortion.getFlatBufferPtr()
  /// is pointed to the corresponding part of this->mFlatBufferPtr
  ///
  TPCDistortionIRS mDistortion;

  bool mApplyDistortion; // flag for applying distortion

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
  /// Since this correction requires a knowledge of the spatial position, it is appied after mDistortion,
  /// not on the drift length but directly on V coordinate.
  ///
  /// mTOFcorr == mVdrift/(speed of light)
  ///
  float mTOFcorr;

  float mPrimVtxZ; ///< Z of the primary vertex, needed for the Time-Of-Flight correction

  ClassDefNV(TPCFastTransform, 1);
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
  convPadTimeToUV(slice, row, pad, time, u, v, maxTimeBin);
  if (slice < getGeometry().getNumberOfSlicesA()) {
    v += getGeometry().getTPCzLengthA();
  } else {
    v += getGeometry().getTPCzLengthC();
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

GPUdi() void TPCFastTransform::convUVtoPadTimeInTimeFrame(int slice, int row, float u, float v, float& pad, float& time, float maxTimeBin) const
{
  if (slice < getGeometry().getNumberOfSlicesA()) {
    v -= getGeometry().getTPCzLengthA();
  } else {
    v -= getGeometry().getTPCzLengthC();
  }
  convUVtoPadTime(slice, row, u, v, pad, time, maxTimeBin);
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

  if (mApplyDistortion) {
    float dx, du, dv;
    mDistortion.getDistortion(slice, row, u, v, dx, du, dv);
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
  /// Distortions and Time-Of-Flight correction are not alpplied.
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
  /// No space charge distortions, no time of flight correction
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);

  x = rowInfo.x;
  float u = (pad - 0.5 * rowInfo.maxPad) * rowInfo.padWidth;
  float v = (time - mT0 - vertexTime) * mVdrift; // drift length cm

  getGeometry().convUVtoLocal(slice, u, v, y, z);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
