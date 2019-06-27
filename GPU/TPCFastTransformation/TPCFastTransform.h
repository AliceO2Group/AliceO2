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
#include "TPCDistortionIRS.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

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
  /// The struct contains necessary info for TPC slice
  struct SliceInfo {
    float sinAlpha;
    float cosAlpha;
  };

  /// The struct contains necessary info for TPC padrow
  struct RowInfo {
    float x;        ///< x coordinate of the row [cm]
    int maxPad;     ///< maximal pad number = n pads - 1
    float padWidth; ///< width of pads [cm]
  };

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
  void startConstruction(int numberOfRows);

  /// Initializes a TPC row
  void setTPCrow(int iRow, float x, int nPads, float padWidth);

  /// Sets TPC geometry
  ///
  /// It must be called once during initialization
  void setTPCgeometry(float tpcZlengthSideA, float tpcZlengthSideC);

  /// Sets all drift calibration parameters and the time stamp
  ///
  /// It must be called once during construction,
  /// but also may be called afterwards to reset these parameters.
  void setCalibration(long int timeStamp, float t0, float vDrift, float vDriftCorrY, float lDriftCorr, float tofCorr, float primVtxZ, float tpcAlignmentZ);

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
  GPUd() int Transform(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime = 0) const;
  GPUd() int TransformInTimeFrame(int slice, int row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const;

  GPUdi() int convLocalToGlobal(int slice, float lx, float ly, float lz, float& gx, float& gy, float& gz);
  GPUdi() int convGlobalToLocal(int slice, float gx, float gy, float gz, float& lx, float& ly, float& lz);

  GPUd() int convPadTimeToUV(int slice, int row, float pad, float time, float& u, float& v, float vertexTime) const;
  GPUd() int convUVtoYZ(int slice, int row, float x, float u, float v, float& y, float& z) const;
  GPUd() int getTOFcorrection(int slice, int row, float x, float y, float z, float& dz) const;

  GPUd() int convYZtoUV(int slice, int row, float x, float y, float z, float& u, float& v) const;
  GPUd() int convUVtoPadTime(int slice, int row, float u, float v, float& pad, float& time) const;

  GPUd() int convPadTimeToUVInTimeFrame(int slice, int row, float pad, float time, float& u, float& v, float maxTimeBin) const;

  void setApplyDistortionFlag(bool flag) { mApplyDistortion = flag; }
  bool getApplyDistortionFlag() { return mApplyDistortion; }

  /// _______________  Utilities  _______________________________________________

  /// Gives number of TPC slices
  static int getNumberOfSlices() { return NumberOfSlices; }

  /// Gives number of TPC rows
  int getNumberOfRows() const { return mNumberOfRows; }

  /// Gives the time stamp of the current calibaration parameters
  long int getTimeStamp() const { return mTimeStamp; }

  /// Gives slice info
  GPUd() const SliceInfo& getSliceInfo(int slice) const { return mSliceInfos[slice]; }

  /// Gives TPC row info
  GPUd() const RowInfo& getRowInfo(int row) const { return mRowInfoPtr[row]; }

  /// Gives Z length of the TPC, side A
  GPUd() float getTPCzLengthA() const { return mTPCzLengthA; }

  /// Gives Z length of the TPC, side C
  GPUd() float getTPCzLengthC() const { return mTPCzLengthC; }

  /// Print method
  void Print() const;

 private:
  /// Enumeration of possible initialization states
  enum ConstructionExtraState : unsigned int {
    GeometryIsSet = 0x4,   ///< the TPC geometry is set
    CalibrationIsSet = 0x8 ///< the drift calibration is set
  };

  /// _______________  Utilities  _______________________________________________

  void relocateBufferPointers(const char* oldBuffer, char* actualBuffer);

  /// _______________  Data members  _______________________________________________

  static constexpr int NumberOfSlices = 36; ///< Number of TPC slices ( slice = inner + outer sector )

  /// _______________  Construction control  _______________________________________________

  int mConstructionCounter;                              ///< counter for initialized parameters
  std::unique_ptr<RowInfo[]> mConstructionRowInfoBuffer; ///< Temporary container of the row infos during initialization

  /// _______________  Geometry  _______________________________________________

  SliceInfo mSliceInfos[NumberOfSlices]; ///< array of slice information [fixed size]

  int mNumberOfRows = 0; ///< Number of TPC rows. It is different for the Run2 and the Run3 setups

  const RowInfo* mRowInfoPtr; ///< pointer to RowInfo array inside the mFlatBufferPtr buffer

  float mTPCzLengthA; ///< Z length of the TPC, side A
  float mTPCzLengthC; ///< Z length of the TPC, side C

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

  float mPrimVtxZ;      ///< Z of the primary vertex, needed for the Time-Of-Flight correction
  float mTPCalignmentZ; ///< Global Z shift of the TPC detector. It is applied at the end of the transformation.
};

// =======================================================================
//              Inline implementations of some methods
// =======================================================================

inline void TPCFastTransform::setTPCrow(int iRow, float x, int nPads, float padWidth)
{
  /// Initializes a TPC row
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(iRow >= 0 && iRow < mNumberOfRows);
  RowInfo& row = mConstructionRowInfoBuffer[iRow];
  row.x = x;
  row.maxPad = nPads - 1;
  row.padWidth = padWidth;
  mConstructionCounter++;
}

GPUdi() int TPCFastTransform::convLocalToGlobal(int slice, float lx, float ly, float lz, float& gx, float& gy, float& gz)
{
  if (slice < 0 || slice >= NumberOfSlices) {
    return -1;
  }
  const SliceInfo& sliceInfo = getSliceInfo(slice);
  gx = lx * sliceInfo.cosAlpha - ly * sliceInfo.sinAlpha;
  gy = lx * sliceInfo.sinAlpha + ly * sliceInfo.cosAlpha;
  gz = lz;
  return 0;
}

GPUdi() int TPCFastTransform::convGlobalToLocal(int slice, float gx, float gy, float gz, float& lx, float& ly, float& lz)
{
  if (slice < 0 || slice >= NumberOfSlices) {
    return -1;
  }
  const SliceInfo& sliceInfo = getSliceInfo(slice);
  lx = gx * sliceInfo.cosAlpha + gy * sliceInfo.sinAlpha;
  ly = -gx * sliceInfo.sinAlpha + gy * sliceInfo.cosAlpha;
  lz = gz;
  return 0;
}

GPUdi() int TPCFastTransform::convPadTimeToUV(int slice, int row, float pad, float time, float& u, float& v, float vertexTime) const
{
  if (slice < 0 || slice >= NumberOfSlices || row < 0 || row >= mNumberOfRows) {
    return -1;
  }

  bool sideC = (slice >= NumberOfSlices / 2);

  const RowInfo& rowInfo = getRowInfo(row);
  const SliceInfo& sliceInfo = getSliceInfo(slice);

  float x = rowInfo.x;
  u = (pad - 0.5 * rowInfo.maxPad) * rowInfo.padWidth;

  float y = sideC ? -u : u; // pads are mirrorred on C-side
  float yLab = y * sliceInfo.cosAlpha + x * sliceInfo.sinAlpha;

  v = (time - mT0 - vertexTime) * (mVdrift + mVdriftCorrY * yLab) + mLdriftCorr; // drift length cm
  return 0;
}

GPUdi() int TPCFastTransform::convPadTimeToUVInTimeFrame(int slice, int row, float pad, float time, float& u, float& v, float maxTimeBin) const
{
  if (slice < 0 || slice >= NumberOfSlices || row < 0 || row >= mNumberOfRows) {
    return -1;
  }

  bool sideC = (slice >= NumberOfSlices / 2);

  const RowInfo& rowInfo = getRowInfo(row);
  const SliceInfo& sliceInfo = getSliceInfo(slice);

  float x = rowInfo.x;
  u = (pad - 0.5 * rowInfo.maxPad) * rowInfo.padWidth;

  float y = sideC ? -u : u; // pads are mirrorred on C-side
  float yLab = y * sliceInfo.cosAlpha + x * sliceInfo.sinAlpha;

  v = (time - mT0 - maxTimeBin) * (mVdrift + mVdriftCorrY * yLab) + mLdriftCorr; // drift length cm

  if (sideC) {
    v += mTPCzLengthC;
  } else {
    v += mTPCzLengthA;
  }

  return 0;
}

GPUdi() int TPCFastTransform::convUVtoPadTime(int slice, int row, float u, float v, float& pad, float& time) const
{
  if (slice < 0 || slice >= NumberOfSlices || row < 0 || row >= mNumberOfRows) {
    return -1;
  }

  bool sideC = (slice >= NumberOfSlices / 2);

  const RowInfo& rowInfo = getRowInfo(row);
  const SliceInfo& sliceInfo = getSliceInfo(slice);

  pad = u / rowInfo.padWidth + 0.5 * rowInfo.maxPad;

  float x = rowInfo.x;
  float y = sideC ? -u : u; // pads are mirrorred on C-side
  float yLab = y * sliceInfo.cosAlpha + x * sliceInfo.sinAlpha;
  time = mT0 + (v - mLdriftCorr) / (mVdrift + mVdriftCorrY * yLab);
  return 0;
}

GPUdi() int TPCFastTransform::convUVtoYZ(int slice, int row, float x, float u, float v, float& y, float& z) const
{
  if (slice < 0 || slice >= NumberOfSlices || row < 0 || row >= mNumberOfRows) {
    return -1;
  }

  bool sideC = (slice >= NumberOfSlices / 2);

  if (sideC) {
    y = -u;               // pads are mirrorred on C-side
    z = v - mTPCzLengthC; // drift direction is mirrored on C-side
  } else {
    y = u;
    z = mTPCzLengthA - v;
  }

  // global TPC alignment
  z += mTPCalignmentZ;
  return 0;
}

GPUdi() int TPCFastTransform::convYZtoUV(int slice, int row, float x, float y, float z, float& u, float& v) const
{
  if (slice < 0 || slice >= NumberOfSlices || row < 0 || row >= mNumberOfRows) {
    return -1;
  }

  bool sideC = (slice >= NumberOfSlices / 2);

  z = z - mTPCalignmentZ;
  if (sideC) {
    u = -y;
    v = z + mTPCzLengthC;
  } else {
    u = y;
    v = mTPCzLengthA - z;
  }
  return 0;
}

GPUdi() int TPCFastTransform::getTOFcorrection(int slice, int row, float x, float y, float z, float& dz) const
{
  // calculate time of flight correction for  z coordinate
  if (slice < 0 || slice >= NumberOfSlices || row < 0 || row >= mNumberOfRows) {
    return -1;
  }
  bool sideC = (slice >= NumberOfSlices / 2);
  float distZ = z - mPrimVtxZ;
  float dv = -sqrt(x * x + y * y + distZ * distZ) * mTOFcorr;
  dz = sideC ? dv : -dv;
  return 0;
}

GPUdi() int TPCFastTransform::Transform(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a slice
  /// taking calibration + alignment into account.
  ///

  if (slice < 0 || slice >= NumberOfSlices || row < 0 || row >= mNumberOfRows) {
    return -1;
  }

  const RowInfo& rowInfo = getRowInfo(row);
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

  convUVtoYZ(slice, row, x, u, v, y, z);

  float dzTOF = 0;
  getTOFcorrection(slice, row, x, y, z, dzTOF);
  z += dzTOF;
  return 0;
}

GPUdi() int TPCFastTransform::TransformInTimeFrame(int slice, int row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Distortions and Time-Of-Flight correction are not alpplied.
  ///

  if (slice < 0 || slice >= NumberOfSlices || row < 0 || row >= mNumberOfRows) {
    return -1;
  }

  const RowInfo& rowInfo = getRowInfo(row);

  x = rowInfo.x;
  float u = 0, v = 0;
  convPadTimeToUVInTimeFrame(slice, row, pad, time, u, v, maxTimeBin);
  convUVtoYZ(slice, row, x, u, v, y, z);
  return 0;
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
