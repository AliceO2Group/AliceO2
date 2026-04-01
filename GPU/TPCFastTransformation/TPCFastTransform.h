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
#include "GPUDebugStreamer.h"

#if !defined(GPUCA_GPUCODE)
#include <string>
#endif // !GPUCA_GPUCODE

namespace o2::tpc
{
template <class T>
class SpaceCharge;
}

namespace o2
{
namespace gpu
{

/// simple struct to hold the space charge object which can be used for CPU reconstruction only
struct TPCSlowSpaceChargeCorrection {

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// destructor
  ~TPCSlowSpaceChargeCorrection();

  /// getting the corrections for global coordinates
  void getCorrections(const float gx, const float gy, const float gz, const int32_t sector, float& gdxC, float& gdyC, float& gdzC) const;

  o2::tpc::SpaceCharge<float>* mCorr{nullptr}; ///< reference space charge corrections
#else
  ~TPCSlowSpaceChargeCorrection() = default;

  /// setting dummy corrections for GPU
  GPUdi() void getCorrections(const float gx, const float gy, const float gz, const int32_t sector, float& gdxC, float& gdyC, float& gdzC) const
  {
    gdxC = 0;
    gdyC = 0;
    gdzC = 0;
  }
#endif

  ClassDefNV(TPCSlowSpaceChargeCorrection, 2);
};

///
/// The TPCFastTransform class represents transformation of raw TPC coordinates to XYZ
///
/// (TPC Row number, Pad, Drift Time) ->  (X,Y,Z)
///
/// The following coordinate systems are used:
///
/// 1. raw coordinate system: TPC row number [int32_t], readout pad number [float], drift time [float]
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
  static constexpr float DEFLUMI = -1e6f; // default value to check if member was set
  static constexpr float DEFIDC = -1e6f;  // default value to check if member was set

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastTransform();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject() instead
  TPCFastTransform(const TPCFastTransform&) = delete;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject() instead
  TPCFastTransform& operator=(const TPCFastTransform&) = delete;

  inline void destroy()
  {
    mCorrection.destroy();
    FlatObject::destroy();
  }

/// Destructor
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && defined(GPUCA_O2_LIB)
  ~TPCFastTransform()
  {
    delete mCorrectionSlow;
  }
#else
  ~TPCFastTransform() = default;
#endif

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
  void setCalibration1(int64_t timeStamp, float t0, float vDrift);

  /// Set Lumi info
  void setLumi(float l) { mLumi = l; }
  void setLumiError(float e) { mLumiError = e; }
  void setLumiScaleFactor(float s) { mLumiScaleFactor = s; }
  void setIDC(float l) { mIDC = l; }
  void setIDCError(float e) { mIDCError = e; }
  void setCTP2IDCFallBackThreshold(float v) { mCTP2IDCFallBackThreshold = v; }
  /// Sets the time stamp of the current calibaration
  void setTimeStamp(int64_t v) { mTimeStamp = v; }

  /// Gives a reference for external initialization of TPC corrections
  GPUdi() const TPCFastSpaceChargeCorrection& getCorrection() const { return mCorrection; }

  /// Gives a reference for external initialization of TPC corrections
  TPCFastSpaceChargeCorrection& getCorrection() { return mCorrection; }

  /// Finishes initialization: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a sector
  /// taking calibration into account.
  ///
  GPUd() void Transform(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime = 0, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;
  GPUd() void TransformXYZ(int32_t sector, int32_t row, float& x, float& y, float& z, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;

  /// Transformation in the time frame
  GPUd() void TransformInTimeFrame(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const;
  GPUd() void TransformInTimeFrame(int32_t sector, float time, float& z, float maxTimeBin) const;

  /// Inverse transformation
  GPUd() void InverseTransformInTimeFrame(int32_t sector, int32_t row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const;
  GPUd() float InverseTransformInTimeFrame(int32_t sector, float z, float maxTimeBin) const;

  /// Inverse transformation: Transformed Y and Z -> transformed X
  GPUd() void InverseTransformYZtoX(int32_t sector, int32_t row, float y, float z, float& x, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;

  /// Inverse transformation: Transformed Y and Z -> Y and Z, transformed w/o space charge correction
  GPUd() void InverseTransformYZtoNominalYZ(int32_t sector, int32_t row, float y, float z, float& ny, float& nz, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;

  /// Inverse transformation: Transformed X, Y and Z -> X, Y and Z, transformed w/o space charge correction
  GPUd() void InverseTransformXYZtoNominalXYZ(int32_t sector, int32_t row, float x, float y, float z, float& nx, float& ny, float& nz, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;

  /// Ideal transformation with Vdrift only - without calibration
  GPUd() void TransformIdeal(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime) const;
  GPUd() void TransformIdealZ(int32_t sector, float time, float& z, float vertexTime) const;

  GPUd() void convPadTimeToLocal(int32_t sector, int32_t row, float pad, float time, float& y, float& z, float vertexTime) const;
  GPUd() void convPadTimeToLocalInTimeFrame(int32_t sector, int32_t row, float pad, float time, float& y, float& z, float maxTimeBin) const;

  GPUd() void convLocalToPadTime(int32_t sector, int32_t row, float y, float z, float& pad, float& time, float vertexTime) const;
  GPUd() void convLocalToPadTimeInTimeFrame(int32_t sector, int32_t row, float y, float z, float& pad, float& time, float maxTimeBin) const;

  GPUd() float convTimeToZinTimeFrame(int32_t sector, float time, float maxTimeBin) const;
  GPUd() float convZtoTimeInTimeFrame(int32_t sector, float z, float maxTimeBin) const;
  GPUd() float convDeltaTimeToDeltaZinTimeFrame(int32_t sector, float deltaTime) const;
  GPUd() float convDeltaZtoDeltaTimeInTimeFrame(int32_t sector, float deltaZ) const;
  GPUd() float convDeltaZtoDeltaTimeInTimeFrameAbs(float deltaZ) const;
  GPUd() float convZOffsetToVertexTime(int32_t sector, float zOffset, float maxTimeBin) const;
  GPUd() float convVertexTimeToZOffset(int32_t sector, float vertexTime, float maxTimeBin) const;

  void setApplyCorrectionOn() { mApplyCorrection = 1; }
  void setApplyCorrectionOff() { mApplyCorrection = 0; }
  bool isCorrectionApplied() { return mApplyCorrection; }

  /// _______________  Utilities  _______________________________________________

  /// TPC geometry information
  GPUdi() const TPCFastTransformGeo& getGeometry() const { return mCorrection.getGeometry(); }

  /// Gives the time stamp of the current calibaration parameters
  GPUdi() int64_t getTimeStamp() const { return mTimeStamp; }

  /// Return mVDrift in cm / time bin
  GPUdi() float getVDrift() const { return mVdrift; }

  /// Return T0 in time bin units
  GPUdi() float getT0() const { return mT0; }

  /// Return map lumi
  GPUdi() float getLumi() const { return mLumi; }

  GPUdi() float isLumiSet() const { return mLumi != DEFLUMI; }

  /// Return map lumi error
  GPUdi() float getLumiError() const { return mLumiError; }

  /// Return map lumi
  GPUd() float getIDC() const;

  GPUdi() bool isIDCSet() const { return mIDC != DEFIDC; }

  /// Return map lumi error
  GPUdi() float getIDCError() const { return mIDCError; }

  GPUdi() float getCTP2IDCFallBackThreshold() const { return mCTP2IDCFallBackThreshold; }

  /// Return map user defined lumi scale factor
  GPUdi() float getLumiScaleFactor() const { return mLumiScaleFactor; }

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int32_t sector, int32_t row, float pad) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int32_t sector, int32_t row) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int32_t sector) const;

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

  int32_t writeToFile(std::string outFName = "", std::string name = "");

  void rectifyAfterReadingFromFile();

  static TPCFastTransform* loadFromFile(std::string inpFName = "", std::string name = "");

  /// setting the reference corrections
  void setSlowTPCSCCorrection(TFile& inpf);

  /// \return returns the space charge object which is used for the slow correction
  const auto& getCorrectionSlow() const { return *mCorrectionSlow; }

#endif // !GPUCA_GPUCODE

  /// Print method
  void print() const;

  GPUd() float convDriftLengthToTime(float driftLength, float vertexTime) const;

 private:
  /// Enumeration of possible initialization states
  enum ConstructionExtraState : uint32_t {
    CalibrationIsSet = 0x4 ///< the drift calibration is set
  };

  /// _______________  Utilities  _______________________________________________

  /// _______________  Data members  _______________________________________________

  /// _______________  Calibration data. See Transform() method  ________________________________

  int64_t mTimeStamp; ///< time stamp of the current calibration

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
  /// L(t,y) = (t-mT0)*mVdrift  ____
  ///
  float mT0;     ///< T0 in [time bin]
  float mVdrift; ///< VDrift in  [cm/time bin]

  float mLumi;            ///< luminosity estimator
  float mLumiError;       ///< error on luminosity
  float mLumiScaleFactor; ///< user correction factor for lumi (e.g. normalization, efficiency correction etc.)

  float mIDC;                      ///< IDC estimator
  float mIDCError;                 ///< error on IDC
  float mCTP2IDCFallBackThreshold; ///< if IDC is not set but requested, use Lumi if it does not exceed this threshold

  /// Correction of (x,u,v) with tricubic interpolator on a regular grid
  TPCSlowSpaceChargeCorrection* mCorrectionSlow{nullptr}; ///< reference space charge corrections

  GPUd() void TransformLocal(int32_t sector, int32_t row, float& x, float& y, float& z, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const;

  ClassDefNV(TPCFastTransform, 5);
};

// =======================================================================
//              Inline implementations of some methods
// =======================================================================

// ----------------------------------------------------------------------

GPUdi() void TPCFastTransform::convPadTimeToLocal(int32_t sector, int32_t row, float pad, float time, float& y, float& z, float vertexTime) const
{
  float l = (time - mT0 - vertexTime) * mVdrift; // drift length [cm]
  getGeometry().convPadDriftLengthToLocal(sector, row, pad, l, y, z);
}

GPUdi() void TPCFastTransform::convPadTimeToLocalInTimeFrame(int32_t sector, int32_t row, float pad, float time, float& y, float& z, float maxTimeBin) const
{
  float l = getGeometry().getTPCzLength() + (time - mT0 - maxTimeBin) * mVdrift; // drift length [cm]
  getGeometry().convPadDriftLengthToLocal(sector, row, pad, l, y, z);
}

// ----------------------------------------------------------------------

GPUdi() float TPCFastTransform::convZOffsetToVertexTime(int32_t sector, float zOffset, float maxTimeBin) const
{
  if (sector < getGeometry().getNumberOfSectorsA()) {
    return maxTimeBin - (getGeometry().getTPCzLength() + zOffset) / mVdrift;
  } else {
    return maxTimeBin - (getGeometry().getTPCzLength() - zOffset) / mVdrift;
  }
}

GPUdi() float TPCFastTransform::convVertexTimeToZOffset(int32_t sector, float vertexTime, float maxTimeBin) const
{
  if (sector < getGeometry().getNumberOfSectorsA()) {
    return (maxTimeBin - vertexTime) * mVdrift - getGeometry().getTPCzLength();
  } else {
    return -((maxTimeBin - vertexTime) * mVdrift - getGeometry().getTPCzLength());
  }
}

GPUdi() float TPCFastTransform::convDriftLengthToTime(float driftLength, float vertexTime) const
{
  return (mT0 + vertexTime + driftLength / mVdrift);
}

// ----------------------------------------------------------------------

GPUdi() void TPCFastTransform::convLocalToPadTime(int32_t sector, int32_t row, float y, float z, float& pad, float& time, float vertexTime) const
{
  float l;
  getGeometry().convLocalToPadDriftLength(sector, row, y, z, pad, l);
  time = convDriftLengthToTime(l, vertexTime);
}

GPUdi() void TPCFastTransform::convLocalToPadTimeInTimeFrame(int32_t sector, int32_t row, float y, float z, float& pad, float& time, float maxTimeBin) const
{
  float l;
  getGeometry().convLocalToPadDriftLength(sector, row, y, z, pad, l);
  time = convDriftLengthToTime(l, maxTimeBin);
}

// ----------------------------------------------------------------------

GPUdi() void TPCFastTransform::TransformLocal(int32_t sector, int32_t row, float& x, float& y, float& z, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{
  GPUCA_RTC_SPECIAL_CODE(ref2 = nullptr; scale2 = 0.f;);

  if (!mApplyCorrection) {
    return;
  }

  float dx = 0.f, dy = 0.f, dz = 0.f;

  if ((scale >= 0.f) || (scaleMode == 1) || (scaleMode == 2)) {
#ifndef GPUCA_GPUCODE
    if (mCorrectionSlow) {
      float gx, gy, gz;
      getGeometry().convLocalToGlobal(sector, x, y, z, gx, gy, gz);
      float gdxC, gdyC, gdzC;
      mCorrectionSlow->getCorrections(gx, gy, gz, sector, gdxC, gdyC, gdzC);
      getGeometry().convGlobalToLocal(sector, gdxC, gdyC, gdzC, dx, dy, dz);
    } else
#endif // GPUCA_GPUCODE
    {
      mCorrection.getCorrectionLocal(sector, row, y, z, dx, dy, dz);
      if (ref) {
        if ((scale > 0.f) && (scaleMode == 0)) { // scaling was requested
          float dx1, dy1, dz1;
          ref->mCorrection.getCorrectionLocal(sector, row, y, z, dx1, dy1, dz1);
          dx = (dx - dx1) * scale + dx1;
          dy = (dy - dy1) * scale + dy1;
          dz = (dz - dz1) * scale + dz1;
        } else if ((scale != 0.f) && ((scaleMode == 1) || (scaleMode == 2))) {
          float dx1, dy1, dz1;
          ref->mCorrection.getCorrectionLocal(sector, row, y, z, dx1, dy1, dz1);
          dx = dx1 * scale + dx;
          dy = dy1 * scale + dy;
          dz = dz1 * scale + dz;
        }
      }
      if (ref2 && (scale2 != 0)) {
        float dx1, dy1, dz1;
        ref2->mCorrection.getCorrectionLocal(sector, row, y, z, dx1, dy1, dz1);
        dx = dx1 * scale2 + dx;
        dy = dy1 * scale2 + dy;
        dz = dz1 * scale2 + dz;
      }
    }
  }

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    float lx = x, ly = y, lz = z;

    float gx, gy, gz;
    getGeometry().convLocalToGlobal(sector, lx, ly, lz, gx, gy, gz);

    float lxT = lx + dx;
    float lyT = ly + dy;
    float lzT = lz + dz;

    float invYZtoXScaled;
    InverseTransformYZtoX(sector, row, lyT, lzT, invYZtoXScaled, ref, ref2, scale, scale2, scaleMode);

    float invYZtoX;
    InverseTransformYZtoX(sector, row, lyT, lzT, invYZtoX);

    float YZtoNominalY;
    float YZtoNominalZ;
    InverseTransformYZtoNominalYZ(sector, row, lyT, lzT, YZtoNominalY, YZtoNominalZ);

    float YZtoNominalYScaled;
    float YZtoNominalZScaled;
    InverseTransformYZtoNominalYZ(sector, row, lyT, lzT, YZtoNominalYScaled, YZtoNominalZScaled, ref, ref2, scale, scale2, scaleMode);

    float dxRef = 0.f, dyRef = 0.f, dzRef = 0.f;
    if (ref) {
      ref->mCorrection.getCorrectionLocal(sector, row, y, z, dxRef, dyRef, dzRef);
    }

    float dxRef2 = 0.f, dyRef2 = 0.f, dzRef2 = 0.f;
    if (ref2) {
      ref2->mCorrection.getCorrectionLocal(sector, row, y, z, dxRef2, dyRef2, dzRef2);
    }

    float dxOrig, dyOrig, dzOrig;
    mCorrection.getCorrectionLocal(sector, row, y, z, dxOrig, dyOrig, dzOrig);

    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_Transform").data()
                                                                                       // corrections in x, u, v
                                                                                       << "dxOrig=" << dxOrig
                                                                                       << "dyOrig=" << dyOrig
                                                                                       << "dzOrig=" << dzOrig
                                                                                       << "dxRef=" << dxRef
                                                                                       << "dyRef=" << dyRef
                                                                                       << "dzRef=" << dzRef
                                                                                       << "dxRef2=" << dxRef2
                                                                                       << "dyRef2=" << dyRef2
                                                                                       << "dzRef2=" << dzRef2
                                                                                       << "dx=" << dx
                                                                                       << "dy=" << dy
                                                                                       << "dz=" << dz
                                                                                       << "row=" << row
                                                                                       << "sector=" << sector
                                                                                       << "scale=" << scale
                                                                                       << "scale2=" << scale2
                                                                                       // original local coordinates
                                                                                       << "ly=" << ly
                                                                                       << "lz=" << lz
                                                                                       << "lx=" << lx
                                                                                       // corrected local coordinated
                                                                                       << "lxT=" << lxT
                                                                                       << "lyT=" << lyT
                                                                                       << "lzT=" << lzT
                                                                                       // global uncorrected coordinates
                                                                                       << "gx=" << gx
                                                                                       << "gy=" << gy
                                                                                       << "gz=" << gz
                                                                                       // some transformations which are applied
                                                                                       << "invYZtoX=" << invYZtoX
                                                                                       << "invYZtoXScaled=" << invYZtoXScaled
                                                                                       << "YZtoNominalY=" << YZtoNominalY
                                                                                       << "YZtoNominalYScaled=" << YZtoNominalYScaled
                                                                                       << "YZtoNominalZ=" << YZtoNominalZ
                                                                                       << "YZtoNominalZScaled=" << YZtoNominalZScaled
                                                                                       << "scaleMode=" << scaleMode
                                                                                       << "\n";
  })

  x += dx;
  y += dy;
  z += dz;
}

GPUdi() void TPCFastTransform::TransformXYZ(int32_t sector, int32_t row, float& x, float& y, float& z, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{

  TransformLocal(sector, row, x, y, z, ref, ref2, scale, scale2, scaleMode);
}

GPUdi() void TPCFastTransform::Transform(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a sector
  /// taking calibration into account.
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);

  x = rowInfo.x;
  convPadTimeToLocal(sector, row, pad, time, y, z, vertexTime);
  TransformLocal(sector, row, x, y, z, ref, ref2, scale, scale2, scaleMode);
}

GPUdi() void TPCFastTransform::TransformInTimeFrame(int32_t sector, float time, float& z, float maxTimeBin) const
{
  float l = (time - mT0 - maxTimeBin) * mVdrift; // drift length cm
  z = getGeometry().convDriftLengthToZ1(sector, l);
}

GPUdi() void TPCFastTransform::TransformInTimeFrame(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Corrections and Time-Of-Flight correction are not alpplied.
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  x = rowInfo.x;
  convPadTimeToLocalInTimeFrame(sector, row, pad, time, y, z, maxTimeBin);
}

GPUdi() void TPCFastTransform::InverseTransformInTimeFrame(int32_t sector, int32_t row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const
{
  /// Inverse transformation to TransformInTimeFrame
  convLocalToPadTimeInTimeFrame(sector, row, y, z, pad, time, maxTimeBin);
}

GPUdi() float TPCFastTransform::InverseTransformInTimeFrame(int32_t sector, float z, float maxTimeBin) const
{
  float pad, time;
  InverseTransformInTimeFrame(sector, 0, 0, 0, z, pad, time, maxTimeBin);
  return time;
}

GPUdi() void TPCFastTransform::TransformIdealZ(int32_t sector, float time, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms time TPC coordinates to local Z withing a sector
  /// Ideal transformation: only Vdrift from DCS.
  /// No space charge corrections, no time of flight correction
  ///

  float l = (time - mT0 - vertexTime) * mVdrift; // drift length cm
  z = getGeometry().convDriftLengthToZ1(sector, l);
}

GPUdi() void TPCFastTransform::TransformIdeal(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a sector
  /// Ideal transformation: only Vdrift from DCS.
  /// No space charge corrections, no time of flight correction
  ///

  x = getGeometry().getRowInfo(row).x;
  float driftLength = (time - mT0 - vertexTime) * mVdrift; // drift length cm
  getGeometry().convPadDriftLengthToLocal(sector, row, pad, driftLength, y, z);
}

GPUdi() float TPCFastTransform::convTimeToZinTimeFrame(int32_t sector, float time, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Corrections and Time-Of-Flight correction are not alpplied.
  /// Only Z coordinate.
  ///

  float v = (time - mT0 - maxTimeBin) * mVdrift; // drift length cm
  float z = (sector < getGeometry().getNumberOfSectorsA()) ? -v : v;
  return z;
}

GPUdi() float TPCFastTransform::convZtoTimeInTimeFrame(int32_t sector, float z, float maxTimeBin) const
{
  /// Inverse transformation of convTimeToZinTimeFrame()
  float v = (sector < getGeometry().getNumberOfSectorsA()) ? -z : z;
  return mT0 + maxTimeBin + v / mVdrift;
}

GPUdi() float TPCFastTransform::convDeltaTimeToDeltaZinTimeFrame(int32_t sector, float deltaTime) const
{
  float deltaZ = deltaTime * mVdrift;
  return sector < getGeometry().getNumberOfSectorsA() ? -deltaZ : deltaZ;
}

GPUdi() float TPCFastTransform::convDeltaZtoDeltaTimeInTimeFrameAbs(float deltaZ) const
{
  return deltaZ / mVdrift;
}

GPUdi() float TPCFastTransform::convDeltaZtoDeltaTimeInTimeFrame(int32_t sector, float deltaZ) const
{
  float deltaT = deltaZ / mVdrift;
  return sector < getGeometry().getNumberOfSectorsA() ? -deltaT : deltaT;
}

GPUdi() float TPCFastTransform::getMaxDriftTime(int32_t sector, int32_t row, float pad) const
{
  /// maximal possible drift time of the active area
  return convDriftLengthToTime(getGeometry().getTPCzLength(), 0.f);
}

GPUdi() float TPCFastTransform::getMaxDriftTime(int32_t sector, int32_t row) const
{
  /// maximal possible drift time of the active area
  return convDriftLengthToTime(getGeometry().getTPCzLength(), 0.f);
}

GPUdi() float TPCFastTransform::getMaxDriftTime(int32_t sector) const
{
  /// maximal possible drift time of the active area
  return convDriftLengthToTime(getGeometry().getTPCzLength(), 0.f);
}

GPUdi() void TPCFastTransform::InverseTransformYZtoX(int32_t sector, int32_t row, float realY, float realZ, float& realX, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{
  GPUCA_RTC_SPECIAL_CODE(ref2 = nullptr; scale2 = 0.f;);
  /// Transformation y,z -> x

  float dx = 0.f;

  if ((scale >= 0.f) || (scaleMode == 1) || (scaleMode == 2)) {
    dx = mCorrection.getCorrectionXatRealYZ(sector, row, realY, realZ);
    if (ref) { // scaling was requested
      if (scaleMode == 0 && scale > 0.f) {
        float dxref = ref->mCorrection.getCorrectionXatRealYZ(sector, row, realY, realZ);
        dx = (dx - dxref) * scale + dxref;
      } else if ((scale != 0) && ((scaleMode == 1) || (scaleMode == 2))) {
        float dxref = ref->mCorrection.getCorrectionXatRealYZ(sector, row, realY, realZ);
        dx = dxref * scale + dx;
      }
    }
    if (ref2 && (scale2 != 0)) {
      float dxref = ref2->mCorrection.getCorrectionXatRealYZ(sector, row, realY, realZ);
      dx = dxref * scale2 + dx;
    }
  }

  realX = mCorrection.getGeometry().getRowInfo(row).x + dx;

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_InverseTransformYZtoX").data()
                                                                                       << "sector=" << sector
                                                                                       << "row=" << row
                                                                                       << "scale=" << scale
                                                                                       << "y=" << realY
                                                                                       << "z=" << realZ
                                                                                       << "x=" << realX
                                                                                       << "\n";
  })
}

GPUdi() void TPCFastTransform::InverseTransformYZtoNominalYZ(int32_t sector, int32_t row, float realY, float realZ, float& measuredY, float& measuredZ, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{
  /// Transformation real y,z -> measured y,z

  GPUCA_RTC_SPECIAL_CODE(ref2 = nullptr; scale2 = 0.f;);

  float dy = 0;
  float dz = 0;

  if ((scale >= 0.f) || (scaleMode == 1) || (scaleMode == 2)) {
    mCorrection.getCorrectionYZatRealYZ(sector, row, realY, realZ, dy, dz);

    if (ref) { // scaling was requested
      if (scaleMode == 0 && scale > 0.f) {
        float dy1, dz1;
        ref->mCorrection.getCorrectionYZatRealYZ(sector, row, realY, realZ, dy1, dz1);
        dy = (dy - dy1) * scale + dy1;
        dz = (dz - dz1) * scale + dz1;
      } else if ((scale != 0) && ((scaleMode == 1) || (scaleMode == 2))) {
        float dy1, dz1;
        ref->mCorrection.getCorrectionYZatRealYZ(sector, row, realY, realZ, dy1, dz1);
        dy = dy1 * scale + dy;
        dz = dz1 * scale + dz;
      }
      if (ref2 && (scale2 != 0)) {
        float dy1, dz1;
        ref2->mCorrection.getCorrectionYZatRealYZ(sector, row, realY, realZ, dy1, dz1);
        dy = dy1 * scale2 + dy;
        dz = dz1 * scale2 + dz;
      }
    }
  }

  measuredY = realY - dy;
  measuredZ = realZ - dz;

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_InverseTransformYZtoNominalYZ").data()
                                                                                       << "sector=" << sector
                                                                                       << "row=" << row
                                                                                       << "scale=" << scale
                                                                                       << "real y=" << realY
                                                                                       << "real z=" << realZ
                                                                                       << "measured y=" << measuredY
                                                                                       << "measured z=" << measuredZ
                                                                                       << "\n";
  })
}

GPUdi() void TPCFastTransform::InverseTransformXYZtoNominalXYZ(int32_t sector, int32_t row, float x, float y, float z, float& nx, float& ny, float& nz, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{
  /// Inverse transformation: Transformed X, Y and Z -> X, Y and Z, transformed w/o space charge correction
  int32_t row2 = row + 1;
  if (row2 >= getGeometry().getNumberOfRows()) {
    row2 = row - 1;
  }
  float nx1, ny1, nz1; // nominal coordinates for row
  float nx2, ny2, nz2; // nominal coordinates for row2
  nx1 = getGeometry().getRowInfo(row).x;
  nx2 = getGeometry().getRowInfo(row2).x;
  InverseTransformYZtoNominalYZ(sector, row, y, z, ny1, nz1, ref, ref2, scale, scale2, scaleMode);
  InverseTransformYZtoNominalYZ(sector, row2, y, z, ny2, nz2, ref, ref2, scale, scale2, scaleMode);
  float c1 = (nx2 - nx) / (nx2 - nx1);
  float c2 = (nx - nx1) / (nx2 - nx1);
  nx = x;
  ny = (ny1 * c1 + ny2 * c2);
  nz = (nz1 * c1 + nz2 * c2);
}

} // namespace gpu
} // namespace o2

#endif
