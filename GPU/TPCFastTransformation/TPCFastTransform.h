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
  void getCorrections(const float gx, const float gy, const float gz, const int32_t roc, float& gdxC, float& gdyC, float& gdzC) const;

  o2::tpc::SpaceCharge<float>* mCorr{nullptr}; ///< reference space charge corrections
#else
  ~TPCSlowSpaceChargeCorrection() = default;

  /// setting dummy corrections for GPU
  GPUd() void getCorrections(const float gx, const float gy, const float gz, const int32_t roc, float& gdxC, float& gdyC, float& gdzC) const
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
  GPUd() const TPCFastSpaceChargeCorrection& getCorrection() const { return mCorrection; }

  /// Gives a reference for external initialization of TPC corrections
  TPCFastSpaceChargeCorrection& getCorrection() { return mCorrection; }

  /// Finishes initialization: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a roc
  /// taking calibration into account.
  ///
  GPUd() void Transform(int32_t roc, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime = 0, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;
  GPUd() void TransformXYZ(int32_t roc, int32_t row, float& x, float& y, float& z, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;

  /// Transformation in the time frame
  GPUd() void TransformInTimeFrame(int32_t roc, int32_t row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const;
  GPUd() void TransformInTimeFrame(int32_t roc, float time, float& z, float maxTimeBin) const;

  /// Inverse transformation
  GPUd() void InverseTransformInTimeFrame(int32_t roc, int32_t row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const;
  GPUd() float InverseTransformInTimeFrame(int32_t roc, float z, float maxTimeBin) const;

  /// Inverse transformation: Transformed Y and Z -> transformed X
  GPUd() void InverseTransformYZtoX(int32_t roc, int32_t row, float y, float z, float& x, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;

  /// Inverse transformation: Transformed Y and Z -> Y and Z, transformed w/o space charge correction
  GPUd() void InverseTransformYZtoNominalYZ(int32_t roc, int32_t row, float y, float z, float& ny, float& nz, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;

  /// Inverse transformation: Transformed X, Y and Z -> X, Y and Z, transformed w/o space charge correction
  GPUd() void InverseTransformXYZtoNominalXYZ(int32_t roc, int32_t row, float x, float y, float z, float& nx, float& ny, float& nz, const TPCFastTransform* ref = nullptr, const TPCFastTransform* ref2 = nullptr, float scale = 0.f, float scale2 = 0.f, int32_t scaleMode = 0) const;

  /// Ideal transformation with Vdrift only - without calibration
  GPUd() void TransformIdeal(int32_t roc, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime) const;
  GPUd() void TransformIdealZ(int32_t roc, float time, float& z, float vertexTime) const;

  GPUd() void convPadTimeToUV(int32_t row, float pad, float time, float& u, float& v, float vertexTime) const;
  GPUd() void convPadTimeToUVinTimeFrame(int32_t row, float pad, float time, float& u, float& v, float maxTimeBin) const;
  GPUd() void convTimeToVinTimeFrame(float time, float& v, float maxTimeBin) const;

  GPUd() void convUVtoPadTime(int32_t row, float u, float v, float& pad, float& time, float vertexTime) const;
  GPUd() void convUVtoPadTimeInTimeFrame(int32_t row, float u, float v, float& pad, float& time, float maxTimeBin) const;
  GPUd() void convVtoTime(float v, float& time, float vertexTime) const;

  GPUd() float convTimeToZinTimeFrame(int32_t roc, float time, float maxTimeBin) const;
  GPUd() float convZtoTimeInTimeFrame(int32_t roc, float z, float maxTimeBin) const;
  GPUd() float convDeltaTimeToDeltaZinTimeFrame(int32_t roc, float deltaTime) const;
  GPUd() float convDeltaZtoDeltaTimeInTimeFrame(int32_t roc, float deltaZ) const;
  GPUd() float convDeltaZtoDeltaTimeInTimeFrameAbs(float deltaZ) const;
  GPUd() float convZOffsetToVertexTime(int32_t sector, float zOffset, float maxTimeBin) const;
  GPUd() float convVertexTimeToZOffset(int32_t sector, float vertexTime, float maxTimeBin) const;

  void setApplyCorrectionOn() { mApplyCorrection = 1; }
  void setApplyCorrectionOff() { mApplyCorrection = 0; }
  bool isCorrectionApplied() { return mApplyCorrection; }

  /// _______________  Utilities  _______________________________________________

  /// TPC geometry information
  GPUd() const TPCFastTransformGeo& getGeometry() const { return mCorrection.getGeometry(); }

  /// Gives the time stamp of the current calibaration parameters
  GPUd() int64_t getTimeStamp() const { return mTimeStamp; }

  /// Return mVDrift in cm / time bin
  GPUd() float getVDrift() const { return mVdrift; }

  /// Return T0 in time bin units
  GPUd() float getT0() const { return mT0; }

  /// Return map lumi
  GPUd() float getLumi() const { return mLumi; }

  GPUd() float isLumiSet() const { return mLumi != DEFLUMI; }

  /// Return map lumi error
  GPUd() float getLumiError() const { return mLumiError; }

  /// Return map lumi
  GPUd() float getIDC() const;

  GPUd() bool isIDCSet() const { return mIDC != DEFIDC; }

  /// Return map lumi error
  GPUd() float getIDCError() const { return mIDCError; }

  GPUd() float getCTP2IDCFallBackThreshold() const { return mCTP2IDCFallBackThreshold; }

  /// Return map user defined lumi scale factor
  GPUd() float getLumiScaleFactor() const { return mLumiScaleFactor; }

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int32_t roc, int32_t row, float pad) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int32_t roc, int32_t row) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int32_t roc) const;

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

  GPUd() void TransformLocal(int32_t roc, int32_t row, float& x, float& y, float& z, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const;

  ClassDefNV(TPCFastTransform, 4);
};

// =======================================================================
//              Inline implementations of some methods
// =======================================================================

GPUdi() void TPCFastTransform::convPadTimeToUV(int32_t row, float pad, float time, float& u, float& v, float vertexTime) const
{
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  float x = rowInfo.x;
  u = (pad - 0.5f * rowInfo.maxPad) * rowInfo.padWidth;
  v = (time - mT0 - vertexTime) * (mVdrift); // drift length cm
}

GPUdi() void TPCFastTransform::convTimeToVinTimeFrame(float time, float& v, float maxTimeBin) const
{
  v = (time - mT0 - maxTimeBin) * mVdrift; // drift length cm
  v += getGeometry().getTPCzLength();
}

GPUdi() void TPCFastTransform::convPadTimeToUVinTimeFrame(int32_t row, float pad, float time, float& u, float& v, float maxTimeBin) const
{
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  u = (pad - 0.5f * rowInfo.maxPad) * rowInfo.padWidth;
  convTimeToVinTimeFrame(time, v, maxTimeBin);
}

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

GPUdi() void TPCFastTransform::convUVtoPadTime(int32_t row, float u, float v, float& pad, float& time, float vertexTime) const
{
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  pad = u / rowInfo.padWidth + 0.5f * rowInfo.maxPad;
  time = mT0 + vertexTime + v / mVdrift;
}

GPUdi() void TPCFastTransform::convVtoTime(float v, float& time, float vertexTime) const
{
  time = mT0 + vertexTime + v / mVdrift;
}

GPUdi() void TPCFastTransform::convUVtoPadTimeInTimeFrame(int32_t row, float u, float v, float& pad, float& time, float maxTimeBin) const
{
  v -= getGeometry().getTPCzLength();
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  pad = u / rowInfo.padWidth + 0.5f * rowInfo.maxPad;
  time = mT0 + maxTimeBin + v / mVdrift;
}

GPUdi() void TPCFastTransform::TransformLocal(int32_t roc, int32_t row, float& x, float& y, float& z, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
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
      getGeometry().convLocalToGlobal(roc, x, y, z, gx, gy, gz);
      float gdxC, gdyC, gdzC;
      mCorrectionSlow->getCorrections(gx, gy, gz, roc, gdxC, gdyC, gdzC);
      getGeometry().convGlobalToLocal(roc, gdxC, gdyC, gdzC, dx, dy, dz);
    } else
#endif // GPUCA_GPUCODE
    {
      std::tie(dx, dy, dz) = mCorrection.getCorrectionLocal(roc, row, y, z);
      if (ref) {
        if ((scale > 0.f) && (scaleMode == 0)) { // scaling was requested
          auto [dxRef, dyRef, dzRef] = ref->mCorrection.getCorrectionLocal(roc, row, y, z);
          dx = (dx - dxRef) * scale + dxRef;
          dy = (dy - dyRef) * scale + dyRef;
          dz = (dz - dzRef) * scale + dzRef;
        } else if ((scale != 0.f) && ((scaleMode == 1) || (scaleMode == 2))) {
          auto [dxRef, dyRef, dzRef] = ref->mCorrection.getCorrectionLocal(roc, row, y, z);
          dx = dxRef * scale + dx;
          dy = dyRef * scale + dy;
          dz = dzRef * scale + dz;
        }
      }
      if (ref2 && (scale2 != 0)) {
        auto [dxRef, dyRef, dzRef] = ref2->mCorrection.getCorrectionLocal(roc, row, y, z);
        dx = dxRef * scale2 + dx;
        dy = dyRef * scale2 + dy;
        dz = dzRef * scale2 + dz;
      }
    }
  }

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    float lx = x, ly = y, lz = z;

    float gx, gy, gz;
    getGeometry().convLocalToGlobal(roc, lx, ly, lz, gx, gy, gz);

    float lxT = lx + dx;
    float lyT = ly + dy;
    float lzT = lz + dz;

    float invYZtoXScaled;
    InverseTransformYZtoX(roc, row, lyT, lzT, invYZtoXScaled, ref, ref2, scale, scale2, scaleMode);

    float invYZtoX;
    InverseTransformYZtoX(roc, row, lyT, lzT, invYZtoX);

    float YZtoNominalY;
    float YZtoNominalZ;
    InverseTransformYZtoNominalYZ(roc, row, lyT, lzT, YZtoNominalY, YZtoNominalZ);

    float YZtoNominalYScaled;
    float YZtoNominalZScaled;
    InverseTransformYZtoNominalYZ(roc, row, lyT, lzT, YZtoNominalYScaled, YZtoNominalZScaled, ref, ref2, scale, scale2, scaleMode);

    float dxRef = 0.f, dyRef = 0.f, dzRef = 0.f;
    if (ref) {
      std::tie(dxRef, dyRef, dzRef) = ref->mCorrection.getCorrectionLocal(roc, row, y, z);
    }

    float dxRef2 = 0.f, duRef2 = 0.f, dvRef2 = 0.f;
    if (ref2) {
      std::tie(dxRef2, duRef2, dvRef2) = ref2->mCorrection.getCorrectionLocal(roc, row, y, z);
    }

    auto [dxOrig, dyOrig, dzOrig] = mCorrection.getCorrectionLocal(roc, row, y, z);

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
                                                                                       << "roc=" << roc
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

GPUdi() void TPCFastTransform::TransformXYZ(int32_t roc, int32_t row, float& x, float& y, float& z, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{

  TransformLocal(roc, row, x, y, z, ref, ref2, scale, scale2, scaleMode);
}

GPUdi() void TPCFastTransform::Transform(int32_t roc, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a roc
  /// taking calibration into account.
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);

  x = rowInfo.x;
  float u = 0, v = 0;
  convPadTimeToUV(row, pad, time, u, v, vertexTime);
  getGeometry().convUVtoLocal(roc, u, v, y, z);

  TransformLocal(roc, row, x, y, z, ref, ref2, scale, scale2, scaleMode);
}

GPUdi() void TPCFastTransform::TransformInTimeFrame(int32_t roc, float time, float& z, float maxTimeBin) const
{
  float v = 0;
  convTimeToVinTimeFrame(time, v, maxTimeBin);
  getGeometry().convVtoLocal(roc, v, z);
}

GPUdi() void TPCFastTransform::TransformInTimeFrame(int32_t roc, int32_t row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Corrections and Time-Of-Flight correction are not alpplied.
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  x = rowInfo.x;
  float u = 0, v = 0;
  convPadTimeToUVinTimeFrame(row, pad, time, u, v, maxTimeBin);
  getGeometry().convUVtoLocal(roc, u, v, y, z);
}

GPUdi() void TPCFastTransform::InverseTransformInTimeFrame(int32_t roc, int32_t row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const
{
  /// Inverse transformation to TransformInTimeFrame
  float u = 0, v = 0;
  getGeometry().convLocalToUV(roc, y, z, u, v);
  convUVtoPadTimeInTimeFrame(row, u, v, pad, time, maxTimeBin);
}

GPUdi() float TPCFastTransform::InverseTransformInTimeFrame(int32_t roc, float z, float maxTimeBin) const
{
  float pad, time;
  InverseTransformInTimeFrame(roc, 0, 0, 0, z, pad, time, maxTimeBin);
  return time;
}

GPUdi() void TPCFastTransform::TransformIdealZ(int32_t roc, float time, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms time TPC coordinates to local Z withing a roc
  /// Ideal transformation: only Vdrift from DCS.
  /// No space charge corrections, no time of flight correction
  ///

  float v = (time - mT0 - vertexTime) * mVdrift; // drift length cm
  getGeometry().convVtoLocal(roc, v, z);
}

GPUdi() void TPCFastTransform::TransformIdeal(int32_t roc, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a roc
  /// Ideal transformation: only Vdrift from DCS.
  /// No space charge corrections, no time of flight correction
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);

  x = rowInfo.x;
  float u = (pad - 0.5f * rowInfo.maxPad) * rowInfo.padWidth;
  float v = (time - mT0 - vertexTime) * mVdrift; // drift length cm

  getGeometry().convUVtoLocal(roc, u, v, y, z);
}

GPUdi() float TPCFastTransform::convTimeToZinTimeFrame(int32_t roc, float time, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Corrections and Time-Of-Flight correction are not alpplied.
  /// Only Z coordinate.
  ///

  float v = (time - mT0 - maxTimeBin) * mVdrift; // drift length cm
  float z = (roc < getGeometry().getNumberOfRocsA()) ? -v : v;
  return z;
}

GPUdi() float TPCFastTransform::convZtoTimeInTimeFrame(int32_t roc, float z, float maxTimeBin) const
{
  /// Inverse transformation of convTimeToZinTimeFrame()
  float v = (roc < getGeometry().getNumberOfRocsA()) ? -z : z;
  return mT0 + maxTimeBin + v / mVdrift;
}

GPUdi() float TPCFastTransform::convDeltaTimeToDeltaZinTimeFrame(int32_t roc, float deltaTime) const
{
  float deltaZ = deltaTime * mVdrift;
  return roc < getGeometry().getNumberOfRocsA() ? -deltaZ : deltaZ;
}

GPUdi() float TPCFastTransform::convDeltaZtoDeltaTimeInTimeFrameAbs(float deltaZ) const
{
  return deltaZ / mVdrift;
}

GPUdi() float TPCFastTransform::convDeltaZtoDeltaTimeInTimeFrame(int32_t roc, float deltaZ) const
{
  float deltaT = deltaZ / mVdrift;
  return roc < getGeometry().getNumberOfRocsA() ? -deltaT : deltaT;
}

/*
GPUdi() float TPCFastTransform::getLastCalibratedTimeBin(int32_t roc) const
{
  /// Return a value of the last timebin where correction map is valid
  float u, v, pad, time;
  getGeometry().convScaledUVtoUV(roc, 0, 0.f, 1.f, u, v);
  convUVtoPadTime(roc, 0, u, v, pad, time, 0);
  return time;
}
*/

GPUdi() float TPCFastTransform::getMaxDriftTime(int32_t roc, int32_t row, float pad) const
{
  /// maximal possible drift time of the active area
  float maxL = mCorrection.getMaxDriftLength(roc, row, pad);
  return mT0 + maxL / mVdrift;
}

GPUdi() float TPCFastTransform::getMaxDriftTime(int32_t roc, int32_t row) const
{
  /// maximal possible drift time of the active area
  float maxL = mCorrection.getMaxDriftLength(roc, row);
  float maxTime = 0.f;
  convVtoTime(maxL, maxTime, 0.f);
  return maxTime;
}

GPUdi() float TPCFastTransform::getMaxDriftTime(int32_t roc) const
{
  /// maximal possible drift time of the active area
  float maxL = mCorrection.getMaxDriftLength(roc);
  float maxTime = 0.f;
  convVtoTime(maxL, maxTime, 0.f);
  return maxTime;
}

GPUdi() void TPCFastTransform::InverseTransformYZtoX(int32_t roc, int32_t row, float y, float z, float& x, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{
  GPUCA_RTC_SPECIAL_CODE(ref2 = nullptr; scale2 = 0.f;);
  /// Transformation y,z -> x
  float u = 0, v = 0;
  getGeometry().convLocalToUV(roc, y, z, u, v);
  if ((scale >= 0.f) || (scaleMode == 1) || (scaleMode == 2)) {
    mCorrection.getCorrectionInvCorrectedX(roc, row, u, v, x);
    if (ref) { // scaling was requested
      if (scaleMode == 0 && scale > 0.f) {
        float xr;
        ref->mCorrection.getCorrectionInvCorrectedX(roc, row, u, v, xr);
        x = (x - xr) * scale + xr;
      } else if ((scale != 0) && ((scaleMode == 1) || (scaleMode == 2))) {
        float xr;
        ref->mCorrection.getCorrectionInvCorrectedX(roc, row, u, v, xr);
        x = (xr - getGeometry().getRowInfo(row).x) * scale + x; // xr=mGeo.getRowInfo(row).x + dx;
      }
    }
    if (ref2 && (scale2 != 0)) {
      float xr;
      ref2->mCorrection.getCorrectionInvCorrectedX(roc, row, u, v, xr);
      x = (xr - getGeometry().getRowInfo(row).x) * scale2 + x; // xr=mGeo.getRowInfo(row).x + dx;
    }
  } else {
    x = mCorrection.getGeometry().getRowInfo(row).x; // corrections are disabled
  }
  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_InverseTransformYZtoX").data()
                                                                                       << "roc=" << roc
                                                                                       << "row=" << row
                                                                                       << "scale=" << scale
                                                                                       << "y=" << y
                                                                                       << "z=" << z
                                                                                       << "x=" << x
                                                                                       << "v=" << v
                                                                                       << "u=" << u
                                                                                       << "\n";
  })
}

GPUdi() void TPCFastTransform::InverseTransformYZtoNominalYZ(int32_t roc, int32_t row, float y, float z, float& ny, float& nz, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
{
  GPUCA_RTC_SPECIAL_CODE(ref2 = nullptr; scale2 = 0.f;);
  /// Transformation y,z -> x
  float u = 0, v = 0, un = 0, vn = 0;
  getGeometry().convLocalToUV(roc, y, z, u, v);
  if ((scale >= 0.f) || (scaleMode == 1) || (scaleMode == 2)) {
    mCorrection.getCorrectionInvUV(roc, row, u, v, un, vn);
    if (ref) { // scaling was requested
      if (scaleMode == 0 && scale > 0.f) {
        float unr = 0, vnr = 0;
        ref->mCorrection.getCorrectionInvUV(roc, row, u, v, unr, vnr);
        un = (un - unr) * scale + unr;
        vn = (vn - vnr) * scale + vnr;
      } else if ((scale != 0) && ((scaleMode == 1) || (scaleMode == 2))) {
        float unr = 0, vnr = 0;
        ref->mCorrection.getCorrectionInvUV(roc, row, u, v, unr, vnr);
        un = (unr - u) * scale + un; // unr = u - duv[0];
        vn = (vnr - v) * scale + vn;
      }
      if (ref2 && (scale2 != 0)) {
        float unr = 0, vnr = 0;
        ref2->mCorrection.getCorrectionInvUV(roc, row, u, v, unr, vnr);
        un = (unr - u) * scale2 + un; // unr = u - duv[0];
        vn = (vnr - v) * scale2 + vn;
      }
    }
  } else {
    un = u;
    vn = v;
  }
  getGeometry().convUVtoLocal(roc, un, vn, ny, nz);

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_InverseTransformYZtoNominalYZ").data()
                                                                                       << "roc=" << roc
                                                                                       << "row=" << row
                                                                                       << "scale=" << scale
                                                                                       << "y=" << y
                                                                                       << "z=" << z
                                                                                       << "ny=" << ny
                                                                                       << "nz=" << nz
                                                                                       << "u=" << u
                                                                                       << "v=" << v
                                                                                       << "un=" << un
                                                                                       << "vn=" << vn
                                                                                       << "\n";
  })
}

GPUdi() void TPCFastTransform::InverseTransformXYZtoNominalXYZ(int32_t roc, int32_t row, float x, float y, float z, float& nx, float& ny, float& nz, const TPCFastTransform* ref, const TPCFastTransform* ref2, float scale, float scale2, int32_t scaleMode) const
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
  InverseTransformYZtoNominalYZ(roc, row, y, z, ny1, nz1, ref, ref2, scale, scale2, scaleMode);
  InverseTransformYZtoNominalYZ(roc, row2, y, z, ny2, nz2, ref, ref2, scale, scale2, scaleMode);
  float c1 = (nx2 - nx) / (nx2 - nx1);
  float c2 = (nx - nx1) / (nx2 - nx1);
  nx = x;
  ny = (ny1 * c1 + ny2 * c2);
  nz = (nz1 * c1 + nz2 * c2);
}

} // namespace gpu
} // namespace o2

#endif
