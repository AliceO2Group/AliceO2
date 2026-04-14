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

/// \file  TPCFastSpaceChargeCorrection.h
/// \brief Definition of TPCFastSpaceChargeCorrection class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTSPACECHARGECORRECTION_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTSPACECHARGECORRECTION_H

#include "Spline2D.h"
#include "TPCFastTransformGeo.h"
#include "FlatObject.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include "GPUCommonArray.h" // Would work on GPU, but yields performance regressions
#endif

namespace o2::gpu
{

///
/// The TPCFastSpaceChargeCorrection class represents correction of nominal coordinates of TPC clusters
/// using best-fit splines
///
/// Row, U, V -> dX,dU,dV
///
/// The class is flat C structure. No virtual methods, no ROOT types are used.
///
class TPCFastSpaceChargeCorrection : public FlatObject
{
  friend class TPCFastTransformPOD;

 public:
  // obsolete structure, declared here only for backward compatibility
  struct SliceInfo {
    ClassDefNV(SliceInfo, 2);
  };

  struct GridInfo {
   private:
    float y0{0.f};                 ///< Y coordinate of the U-grid start
    float yScale{0.f};             //< scale Y to U-grid coordinate
    float z0{0.f};                 ///< Z coordinate of the V-grid start
    float zScale{0.f};             //< scale Z to V-grid coordinate
    float zOut{0.f};               // outer z of the grid;
    float splineScalingWithZ{0.f}; ///< spline scaling factor in the Z region between the zOut and the readout plane

   public:
    void set(float y0_, float yScale_, float z0_, float zScale_, float zOut_, float zReadout_)
    {
      this->y0 = y0_;
      this->yScale = yScale_;
      this->z0 = z0_;
      this->zScale = zScale_;
      this->zOut = zOut_;
      // no scaling when the distance to the readout is too small
      this->splineScalingWithZ = fabs(zReadout_ - zOut_) > 1.f ? 1.f / (zReadout_ - zOut_) : 0.f;
    }

    float getY0() const { return y0; }
    float getYscale() const { return yScale; }
    float getZ0() const { return z0; }
    float getZscale() const { return zScale; }

    GPUdi() float getSpineScaleForZ(float z) const
    {
      return 1.f - GPUCommonMath::Clamp((z - zOut) * splineScalingWithZ, 0.f, 1.f);
    }

    /// convert local y, z to internal grid coordinates u,v, and spline scale
    GPUdi() void convLocalToGridUntruncated(float y, float z, float& u, float& v, float& s) const
    {
      u = (y - y0) * yScale;
      v = (z - z0) * zScale;
      s = getSpineScaleForZ(z);
    }

    /// convert internal grid coordinates u,v to local y, z
    GPUdi() void convGridToLocal(float gridU, float gridV, float& y, float& z) const
    {
      y = y0 + gridU / yScale;
      z = z0 + gridV / zScale;
    }
    ClassDefNV(GridInfo, 1);
  };

  struct SectorRowInfo {
    int32_t splineScenarioID{0};  ///< scenario index (which of Spline2D splines to use)
    size_t dataOffsetBytes[3]{0}; ///< offset for the spline data withing a TPC sector

    GridInfo gridMeasured; ///< grid info for measured coordinates
    GridInfo gridReal;     ///< grid info for real coordinates

    ClassDefNV(SectorRowInfo, 2);
  };

  typedef Spline2D<float, 3> SplineTypeXYZ;
  typedef Spline2D<float, 1> SplineTypeInvX;
  typedef Spline2D<float, 2> SplineTypeInvYZ;

  typedef SplineTypeXYZ SplineType;

  /// Slim variants (NoFlatObject base) for use in TPCFastTransformPOD
  using SlimSplineTypeXYZ = Spline2D<float, 3, NoFlatObject>;
  using SlimSplineTypeInvX = Spline2D<float, 1, NoFlatObject>;
  using SlimSplineTypeInvYZ = Spline2D<float, 2, NoFlatObject>;
  using SlimSplineType = SlimSplineTypeXYZ;

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastSpaceChargeCorrection();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCFastSpaceChargeCorrection(const TPCFastSpaceChargeCorrection&) = delete;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCFastSpaceChargeCorrection& operator=(const TPCFastSpaceChargeCorrection&) = delete;

  /// Destructor
  ~TPCFastSpaceChargeCorrection();

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Memory alignment

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

  /// Construction interface

  void cloneFromObject(const TPCFastSpaceChargeCorrection& obj, char* newFlatBufferPtr);
  void destroy();

  /// Making the data buffer external

  using FlatObject::releaseInternalBuffer;
  void moveBufferTo(char* newBufferPtr);

  /// Moving the class with its external buffer to another location

  void setActualBufferAddressOld(char* actualFlatBufferPtr);
  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

  /// _______________  Construction interface  ________________________

  /// Starts the construction procedure, reserves temporary memory
  void startConstruction(const TPCFastTransformGeo& geo, int32_t numberOfSplineScenarios);

  /// Initializes a TPC row
  void setRowScenarioID(int32_t iSector, int32_t iRow, int32_t iScenario);

  /// Sets approximation scenario
  void setSplineScenario(int32_t scenarioIndex, const SplineType& spline);

  /// Finishes construction: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  void constructWithNoCorrection(const TPCFastTransformGeo& geo);

  /// _______________  Initialization interface  ________________________

  /// Set no correction
  GPUd() void setNoCorrection();

  /// Sets the time stamp of the current calibaration
  GPUdi() void setTimeStamp(int64_t v) { mTimeStamp = v; }

  /// Gives const pointer to a spline
  GPUd() const SplineType& getSpline(int32_t sector, int32_t row) const;

  /// Gives pointer to a spline
  GPUd() SplineType& getSpline(int32_t sector, int32_t row);

  /// Gives pointer to spline data
  GPUd() float* getCorrectionData(int32_t sector, int32_t row, int32_t iSpline = 0);

  /// Gives pointer to spline data
  GPUd() const float* getCorrectionData(int32_t sector, int32_t row, int32_t iSpline = 0) const;

  /// Gives const pointer to a spline for the inverse X correction
  GPUd() const SplineTypeInvX& getSplineInvX(int32_t sector, int32_t row) const;

  /// Gives pointer to a spline for the inverse X correction
  GPUd() SplineTypeInvX& getSplineInvX(int32_t sector, int32_t row);

  /// Gives pointer to spline data for the inverse X correction
  GPUd() float* getCorrectionDataInvX(int32_t sector, int32_t row);

  /// Gives pointer to spline data for the inverse X correction
  GPUd() const float* getCorrectionDataInvX(int32_t sector, int32_t row) const;

  /// Gives const pointer to a spline for the inverse YZ correction
  GPUd() const SplineTypeInvYZ& getSplineInvYZ(int32_t sector, int32_t row) const;

  /// Gives pointer to a spline for the inverse YZ correction
  GPUd() SplineTypeInvYZ& getSplineInvYZ(int32_t sector, int32_t row);

  /// Gives pointer to spline data for the inverse YZ correction
  GPUd() float* getCorrectionDataInvYZ(int32_t sector, int32_t row);

  /// Gives pointer to spline data for the inverse YZ correction
  GPUd() const float* getCorrectionDataInvYZ(int32_t sector, int32_t row) const;

  /// _______________ The main method: cluster correction  _______________________
  ///
  // GPUd() int32_t getCorrectionInternal(int32_t sector, int32_t row, float u, float v, float& dx, float& du, float& dv) const;

  GPUdi() void getCorrectionLocal(int32_t sector, int32_t row, float y, float z, float& dx, float& dy, float& dz) const;

  /// inverse correction: Real Y and Z -> Real X
  GPUd() float getCorrectionXatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const;

  /// inverse correction: Real Y and Z -> measred Y and Z
  GPUd() void getCorrectionYZatRealYZ(int32_t sector, int32_t row, float realY, float realZ, float& y, float& z) const;

  /// _______________  Utilities  _______________________________________________

  /// convert local y, z to internal grid coordinates u,v
  /// return values: u, v, scaling factor
  GPUd() void convLocalToGrid(int32_t sector, int32_t row, float y, float z, float& u, float& v, float& s) const;

  /// convert internal grid coordinates u,v to local y, z
  /// return values: y, z, scaling factor
  GPUd() void convGridToLocal(int32_t sector, int32_t row, float u, float v, float& y, float& z) const;

  /// convert real Y, Z to the internal grid coordinates
  /// return values: u, v, scaling factor
  GPUd() void convRealLocalToGrid(int32_t sector, int32_t row, float y, float z, float& u, float& v, float& s) const;

  /// convert internal grid coordinates to the real Y, Z
  /// return values: y, z
  GPUd() void convGridToRealLocal(int32_t sector, int32_t row, float u, float v, float& y, float& z) const;

  GPUd() bool isLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const;
  GPUd() bool isRealLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const;

  /// TPC geometry information
  GPUdi() const TPCFastTransformGeo& getGeometry() const
  {
    return mGeo;
  }

  /// Gives the time stamp of the current calibaration parameters
  int64_t getTimeStamp() const { return mTimeStamp; }

  /// Gives TPC sector & row info
  GPUdi() const SectorRowInfo& getSectorRowInfo(int32_t sector, int32_t row) const
  {
    return mSectorRowInfos[mGeo.getMaxNumberOfRows() * sector + row];
  }

  /// Gives TPC sector & row info
  GPUdi() SectorRowInfo& getSectorRowInfo(int32_t sector, int32_t row)
  {
    return mSectorRowInfos[mGeo.getMaxNumberOfRows() * sector + row];
  }

#if !defined(GPUCA_GPUCODE)
  /// Print method
  void print() const;
  GPUh() double testInverse(bool prn = 0);
#endif

 private:
  /// relocate buffer pointers
  void relocateBufferPointers(const char* oldBuffer, char* newBuffer);
  /// release temporary memory used during construction
  void releaseConstructionMemory();

  static constexpr float kMaxCorrection = 100.f; ///< maximum correction value, used to protect from FPEs

  /// _______________  Data members  _______________________________________________

  /// _______________  Construction control  _______________________________________________

  SplineType* mConstructionScenarios = nullptr; //! (transient!!) Temporary container for spline scenarios

  /// _______________  Geometry  _______________________________________________

  TPCFastTransformGeo mGeo; ///< TPC geometry information

  int32_t mNumberOfScenarios; ///< Number of approximation spline scenarios

  SplineType* mScenarioPtr; //! (transient!!) pointer to spline scenarios

  /// _______________  Calibration data  _______________________________________________

  int64_t mTimeStamp; ///< time stamp of the current calibration

  char* mCorrectionData[3]; //! (transient!!) pointer to the spline data in the flat buffer

  size_t mCorrectionDataSize[3]; ///< size of the data per transformation (direct, inverseX, inverse YZ) in the flat buffer

  /// Class version. It is used to read older versions from disc.
  /// The default version 3 is the one before this field was introduced.
  /// The actual version must be set in startConstruction().
  int32_t mClassVersion{3};

  SectorRowInfo mSectorRowInfos[TPCFastTransformGeo::getNumberOfSectors() * TPCFastTransformGeo::getMaxNumberOfRows()]; ///< SectorRowInfo array

  ClassDefNV(TPCFastSpaceChargeCorrection, 4);
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

GPUdi() const TPCFastSpaceChargeCorrection::SplineType& TPCFastSpaceChargeCorrection::getSpline(int32_t sector, int32_t row) const
{
  /// Gives const pointer to spline
  return mScenarioPtr[getSectorRowInfo(sector, row).splineScenarioID];
}

GPUdi() TPCFastSpaceChargeCorrection::SplineType& TPCFastSpaceChargeCorrection::getSpline(int32_t sector, int32_t row)
{
  /// Gives pointer to spline
  return mScenarioPtr[getSectorRowInfo(sector, row).splineScenarioID];
}

GPUdi() float* TPCFastSpaceChargeCorrection::getCorrectionData(int32_t sector, int32_t row, int32_t iSpline)
{
  /// Gives pointer to spline data
  return reinterpret_cast<float*>(mCorrectionData[iSpline] + getSectorRowInfo(sector, row).dataOffsetBytes[iSpline]);
}

GPUdi() const float* TPCFastSpaceChargeCorrection::getCorrectionData(int32_t sector, int32_t row, int32_t iSpline) const
{
  /// Gives pointer to spline data
  return reinterpret_cast<const float*>(mCorrectionData[iSpline] + getSectorRowInfo(sector, row).dataOffsetBytes[iSpline]);
}

GPUdi() TPCFastSpaceChargeCorrection::SplineTypeInvX& TPCFastSpaceChargeCorrection::getSplineInvX(int32_t sector, int32_t row)
{
  /// Gives pointer to spline for the inverse X correction
  return reinterpret_cast<SplineTypeInvX&>(getSpline(sector, row));
}

GPUdi() const TPCFastSpaceChargeCorrection::SplineTypeInvX& TPCFastSpaceChargeCorrection::getSplineInvX(int32_t sector, int32_t row) const
{
  /// Gives const pointer to spline for the inverse X correction
  return reinterpret_cast<const SplineTypeInvX&>(getSpline(sector, row));
}

GPUdi() float* TPCFastSpaceChargeCorrection::getCorrectionDataInvX(int32_t sector, int32_t row)
{
  /// Gives pointer to spline data for the inverse X correction
  return getCorrectionData(sector, row, 1);
}

GPUdi() const float* TPCFastSpaceChargeCorrection::getCorrectionDataInvX(int32_t sector, int32_t row) const
{
  /// Gives pointer to spline data for the inverse X correction
  return getCorrectionData(sector, row, 1);
}

GPUdi() TPCFastSpaceChargeCorrection::SplineTypeInvYZ& TPCFastSpaceChargeCorrection::getSplineInvYZ(int32_t sector, int32_t row)
{
  /// Gives pointer to spline for the inverse YZ correction
  return reinterpret_cast<SplineTypeInvYZ&>(getSpline(sector, row));
}

GPUdi() const TPCFastSpaceChargeCorrection::SplineTypeInvYZ& TPCFastSpaceChargeCorrection::getSplineInvYZ(int32_t sector, int32_t row) const
{
  /// Gives const pointer to spline for the inverse YZ correction
  return reinterpret_cast<const SplineTypeInvYZ&>(getSpline(sector, row));
}

GPUdi() float* TPCFastSpaceChargeCorrection::getCorrectionDataInvYZ(int32_t sector, int32_t row)
{
  /// Gives pointer to spline data for the inverse YZ correction
  return getCorrectionData(sector, row, 2);
}

GPUdi() const float* TPCFastSpaceChargeCorrection::getCorrectionDataInvYZ(int32_t sector, int32_t row) const
{
  /// Gives pointer to spline data for the inverse YZ correction
  return getCorrectionData(sector, row, 2);
}

GPUdi() void TPCFastSpaceChargeCorrection::convLocalToGrid(int32_t sector, int32_t row, float y, float z, float& u, float& v, float& s) const
{
  /// convert local y, z to internal grid coordinates u,v
  /// return values: u, v, scaling factor
  const SplineType& spline = getSpline(sector, row);
  getSectorRowInfo(sector, row).gridMeasured.convLocalToGridUntruncated(y, z, u, v, s);
  // shrink to the grid
  u = GPUCommonMath::Clamp(u, 0.f, (float)spline.getGridX1().getUmax());
  v = GPUCommonMath::Clamp(v, 0.f, (float)spline.getGridX2().getUmax());
}

GPUdi() bool TPCFastSpaceChargeCorrection::isLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// check if local y, z are inside the grid
  float u, v, s;
  getSectorRowInfo(sector, row).gridMeasured.convLocalToGridUntruncated(y, z, u, v, s);
  const auto& spline = getSpline(sector, row);
  // shrink to the grid
  if (u < 0.f || u > (float)spline.getGridX1().getUmax() || //
      v < 0.f || v > (float)spline.getGridX2().getUmax()) {
    return false;
  }
  return true;
}

GPUdi() bool TPCFastSpaceChargeCorrection::isRealLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// check if local y, z are inside the grid
  float u, v, s;
  getSectorRowInfo(sector, row).gridReal.convLocalToGridUntruncated(y, z, u, v, s);
  const auto& spline = getSpline(sector, row);
  // shrink to the grid
  if (u < 0.f || u > (float)spline.getGridX1().getUmax() || //
      v < 0.f || v > (float)spline.getGridX2().getUmax()) {
    return false;
  }
  return true;
}

GPUdi() void TPCFastSpaceChargeCorrection::convGridToLocal(int32_t sector, int32_t row, float gridU, float gridV, float& y, float& z) const
{
  /// convert internal grid coordinates u,v to local y, z
  getSectorRowInfo(sector, row).gridMeasured.convGridToLocal(gridU, gridV, y, z);
}

GPUdi() void TPCFastSpaceChargeCorrection::convRealLocalToGrid(int32_t sector, int32_t row, float y, float z, float& u, float& v, float& s) const
{
  /// convert real y, z to the internal grid coordinates + scale
  const SplineType& spline = getSpline(sector, row);
  getSectorRowInfo(sector, row).gridReal.convLocalToGridUntruncated(y, z, u, v, s);
  // shrink to the grid
  u = GPUCommonMath::Clamp(u, 0.f, (float)spline.getGridX1().getUmax());
  v = GPUCommonMath::Clamp(v, 0.f, (float)spline.getGridX2().getUmax());
}

GPUdi() void TPCFastSpaceChargeCorrection::convGridToRealLocal(int32_t sector, int32_t row, float gridU, float gridV, float& y, float& z) const
{
  /// convert internal grid coordinates u,v to the real y, z
  getSectorRowInfo(sector, row).gridReal.convGridToLocal(gridU, gridV, y, z);
}

GPUdi() void TPCFastSpaceChargeCorrection::getCorrectionLocal(int32_t sector, int32_t row, float y, float z, float& dx, float& dy, float& dz) const
{
  const auto& info = getSectorRowInfo(sector, row);
  const SplineType& spline = getSpline(sector, row);
  const float* splineData = getCorrectionData(sector, row);

  float u, v, s;
  convLocalToGrid(sector, row, y, z, u, v, s);

  float dxyz[3];
  spline.interpolateAtU(splineData, u, v, dxyz);

  if (CAMath::Abs(dxyz[0]) > kMaxCorrection || CAMath::Abs(dxyz[1]) > kMaxCorrection || CAMath::Abs(dxyz[2]) > kMaxCorrection) {
    s = 0.f; // TODO: DR: Protect from FPEs, fix upstream and remove once guaranteed that it is fixed
  }

  dx = s * dxyz[0];
  dy = s * dxyz[1];
  dz = s * dxyz[2];
}

GPUdi() float TPCFastSpaceChargeCorrection::getCorrectionXatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const
{
  const auto& info = getSectorRowInfo(sector, row);
  float u, v, s;
  convRealLocalToGrid(sector, row, realY, realZ, u, v, s);
  float dx = 0;
  getSplineInvX(sector, row).interpolateAtU(getCorrectionDataInvX(sector, row), u, v, &dx);
  if (CAMath::Abs(dx) > kMaxCorrection) {
    s = 0.f; // TODO: DR: Protect from FPEs, fix upstream and remove once guaranteed that it is fixed
  }
  return s * dx;
}

GPUdi() void TPCFastSpaceChargeCorrection::getCorrectionYZatRealYZ(int32_t sector, int32_t row, float realY, float realZ, float& y, float& z) const
{
  float u, v, s;
  convRealLocalToGrid(sector, row, realY, realZ, u, v, s);
  const auto& info = getSectorRowInfo(sector, row);
  float dyz[2];
  getSplineInvYZ(sector, row).interpolateAtU(getCorrectionDataInvYZ(sector, row), u, v, dyz);
  if (CAMath::Abs(dyz[0]) > kMaxCorrection || CAMath::Abs(dyz[1]) > kMaxCorrection) {
    s = 0.f; // TODO: DR: Protect from FPEs, fix upstream and remove once guaranteed that it is fixed
  }
  y = s * dyz[0];
  z = s * dyz[1];
}

} // namespace o2::gpu

#endif
