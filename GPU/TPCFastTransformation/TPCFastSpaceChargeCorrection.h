// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace GPUCA_NAMESPACE
{
namespace gpu
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
 public:
  ///
  /// \brief The struct contains necessary info for TPC padrow
  ///
  struct RowInfo {
    int splineScenarioID;      ///< scenario index (which of Spline2D splines to use)
    size_t dataOffsetBytes[3]; ///< offset for the spline data withing a TPC slice
  };

  struct SliceRowInfo {
    float CorrU0;
    float scaleCorrUtoGrid;
    float scaleCorrVtoGrid;
  };

  typedef Spline2D<float, 3, 0> SplineType;

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastSpaceChargeCorrection();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCFastSpaceChargeCorrection(const TPCFastSpaceChargeCorrection&) CON_DELETE;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCFastSpaceChargeCorrection& operator=(const TPCFastSpaceChargeCorrection&) CON_DELETE;

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

  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

  /// _______________  Construction interface  ________________________

  /// Starts the construction procedure, reserves temporary memory
  void startConstruction(const TPCFastTransformGeo& geo, int numberOfSplineScenarios);

  /// Initializes a TPC row
  void setRowScenarioID(int iRow, int iScenario);

  /// Sets approximation scenario
  void setSplineScenario(int scenarioIndex, const SplineType& spline);

  /// Finishes construction: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  /// _______________  Initialization interface  ________________________

  /// Sets the time stamp of the current calibaration
  void setTimeStamp(long int v) { mTimeStamp = v; }

  /// Gives pointer to a spline
  GPUd() const SplineType& getSpline(int slice, int row) const;

  /// Gives pointer to spline data
  GPUd() float* getSplineData(int slice, int row, int iSpline = 0);

  /// Gives pointer to spline data
  GPUd() const float* getSplineData(int slice, int row, int iSpline = 0) const;

#if !defined(GPUCA_GPUCODE)
  /// Initialise inverse transformations
  GPUd() void initInverse();
#endif

  /// _______________ The main method: cluster correction  _______________________
  ///
  GPUd() int getCorrection(int slice, int row, float u, float v, float& dx, float& du, float& dv) const;

  GPUd() void getCorrectionInvCorrectedX(int slice, int row, float corrU, float corrV, float& corrX) const;

  GPUd() void getCorrectionInvUV(int slice, int row, float corrU, float corrV, float& nomU, float& nomV) const;

  /// _______________  Utilities  _______________________________________________

  /// TPC geometry information
  GPUd() const TPCFastTransformGeo& getGeometry() const { return mGeo; }

  /// Gives the time stamp of the current calibaration parameters
  long int getTimeStamp() const { return mTimeStamp; }

  /// Gives TPC row info
  GPUd() const RowInfo& getRowInfo(int row) const { return mRowInfoPtr[row]; }

  /// Gives TPC slice&row info
  GPUd() const SliceRowInfo& getSliceRowInfo(int slice, int row) const
  {
    return mSliceRowInfoPtr[mGeo.getNumberOfRows() * slice + row];
  }

#if !defined(GPUCA_GPUCODE)
  /// Print method
  void print() const;
#endif

 private:
  /// relocate buffer pointers
  void relocateBufferPointers(const char* oldBuffer, char* newBuffer);
  /// release temporary memory used during construction
  void releaseConstructionMemory();

  /// _______________  Data members  _______________________________________________

  /// _______________  Construction control  _______________________________________________

  RowInfo* mConstructionRowInfos = nullptr;     //! (transient!!) Temporary container of the row infos during construction
  SplineType* mConstructionScenarios = nullptr; //! (transient!!) Temporary container for spline scenarios

  /// _______________  Geometry  _______________________________________________

  TPCFastTransformGeo mGeo; ///< TPC geometry information

  int mNumberOfScenarios; ///< Number of approximation spline scenarios

  SplineType* mScenarioPtr; //! (transient!!) pointer to spline scenarios

  RowInfo* mRowInfoPtr;           //! (transient!!) pointer to RowInfo array inside the mFlatBufferPtr buffer
  SliceRowInfo* mSliceRowInfoPtr; //! (transient!!) pointer to SliceRowInfo array inside the mFlatBufferPtr

  /// _______________  Calibration data  _______________________________________________

  long int mTimeStamp; ///< time stamp of the current calibration

  char* mSplineData[3]; //! (transient!!) pointer to the spline data in the flat buffer

  size_t mSliceDataSizeBytes[3]; ///< size of the data for one slice in the flat buffer
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

GPUdi() const TPCFastSpaceChargeCorrection::SplineType& TPCFastSpaceChargeCorrection::getSpline(int slice, int row) const
{
  /// Gives pointer to spline
  const RowInfo& rowInfo = mRowInfoPtr[row];
  return mScenarioPtr[rowInfo.splineScenarioID];
}

GPUdi() float* TPCFastSpaceChargeCorrection::getSplineData(int slice, int row, int iSpline)
{
  /// Gives pointer to spline data
  const RowInfo& rowInfo = mRowInfoPtr[row];
  return reinterpret_cast<float*>(mSplineData[iSpline] + mSliceDataSizeBytes[iSpline] * slice + rowInfo.dataOffsetBytes[iSpline]);
}

GPUdi() const float* TPCFastSpaceChargeCorrection::getSplineData(int slice, int row, int iSpline) const
{
  /// Gives pointer to spline data
  const RowInfo& rowInfo = mRowInfoPtr[row];
  return reinterpret_cast<float*>(mSplineData[iSpline] + mSliceDataSizeBytes[iSpline] * slice + rowInfo.dataOffsetBytes[iSpline]);
}

GPUdi() int TPCFastSpaceChargeCorrection::getCorrection(int slice, int row, float u, float v, float& dx, float& du, float& dv) const
{
  const SplineType& spline = getSpline(slice, row);
  const float* splineData = getSplineData(slice, row);
  float su = 0, sv = 0;
  mGeo.convUVtoScaledUV(slice, row, u, v, su, sv);
  su *= spline.getGridU1().getUmax();
  sv *= spline.getGridU2().getUmax();
  float dxuv[3];
  spline.interpolateU(splineData, su, sv, dxuv);
  dx = dxuv[0];
  du = dxuv[1];
  dv = dxuv[2];
  return 0;
}

GPUdi() void TPCFastSpaceChargeCorrection::getCorrectionInvCorrectedX(
  int slice, int row, float u, float v, float& x) const
{
  const RowInfo& rowInfo = getRowInfo(row);
  const SliceRowInfo& sliceRowInfo = getSliceRowInfo(slice, row);
  const Spline2D<float, 1, 0>& spline = reinterpret_cast<const Spline2D<float, 1, 0>&>(getSpline(slice, row));
  const float* splineData = getSplineData(slice, row, 1);
  float gridU = (u - sliceRowInfo.CorrU0) * sliceRowInfo.scaleCorrUtoGrid;
  float gridV = v * sliceRowInfo.scaleCorrVtoGrid;
  spline.interpolateU(splineData, gridU, gridV, &x);
}

GPUdi() void TPCFastSpaceChargeCorrection::getCorrectionInvUV(
  int slice, int row, float corrU, float corrV, float& nomU, float& nomV) const
{
  const RowInfo& rowInfo = getRowInfo(row);
  const SliceRowInfo& sliceRowInfo = getSliceRowInfo(slice, row);
  const Spline2D<float, 2, 0>& spline = reinterpret_cast<const Spline2D<float, 2, 0>&>(getSpline(slice, row));
  const float* splineData = getSplineData(slice, row, 2);
  float gridU = (corrU - sliceRowInfo.CorrU0) * sliceRowInfo.scaleCorrUtoGrid;
  float gridV = corrV * sliceRowInfo.scaleCorrVtoGrid;
  float uv[2];
  spline.interpolateU(splineData, gridU, gridV, uv);
  nomU = uv[0];
  nomV = uv[1];
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
