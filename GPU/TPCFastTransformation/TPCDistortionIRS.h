// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCDistortionIRS.h
/// \brief Definition of TPCDistortionIRS class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCDISTORTIONIRS_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCDISTORTIONIRS_H

#include "IrregularSpline2D3D.h"
#include "TPCFastTransformGeo.h"
#include "FlatObject.h"
#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The TPCDistortionIRS class represents correction of nominal coordinates of TPC clusters
/// using irregular splines
///
/// Row, U, V -> dX,dU,dV
///
/// The class is flat C structure. No virtual methods, no ROOT types are used.
///
class TPCDistortionIRS : public FlatObject
{
 public:
  ///
  /// \brief The struct contains necessary info for TPC padrow
  ///
  struct RowSplineInfo {
    int splineScenarioID;   ///< scenario index (which of IrregularSpline2D3D splines to use)
    size_t dataOffsetBytes; ///< offset for the spline data withing a TPC slice
  };

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCDistortionIRS();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCDistortionIRS(const TPCDistortionIRS&) CON_DELETE;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCDistortionIRS& operator=(const TPCDistortionIRS&) CON_DELETE;

  /// Destructor
  ~TPCDistortionIRS();

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Memory alignment

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

  /// Construction interface

  void cloneFromObject(const TPCDistortionIRS& obj, char* newFlatBufferPtr);
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
  void setSplineScenario(int scenarioIndex, const IrregularSpline2D3D& spline);

  /// Finishes construction: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  /// _______________  Initialization interface  ________________________

  /// Sets the time stamp of the current calibaration
  void setTimeStamp(long int v) { mTimeStamp = v; }

  /// Gives pointer to a spline
  GPUd() const IrregularSpline2D3D& getSpline(int slice, int row) const;

  /// Gives pointer to spline data
  GPUd() float* getSplineDataNonConst(int slice, int row);

  /// Gives pointer to spline data
  GPUd() const float* getSplineData(int slice, int row) const;

  /// _______________ The main method: cluster distortion  _______________________
  ///
  GPUd() int getDistortion(int slice, int row, float u, float v, float& dx, float& du, float& dv) const;

  /// _______________  Utilities  _______________________________________________

  /// TPC geometry information
  GPUd() const TPCFastTransformGeo& getGeometry() const { return mGeo; }

  /// Gives the time stamp of the current calibaration parameters
  long int getTimeStamp() const { return mTimeStamp; }

  /// Gives TPC row info
  GPUd() const RowSplineInfo& getRowSplineInfo(int row) const { return mRowSplineInfoPtr[row]; }

  /// Print method
  void print() const;

 private:
  /// relocate buffer pointers
  void relocateBufferPointers(const char* oldBuffer, char* newBuffer);
  /// release temporary memory used during construction
  void releaseConstructionMemory();

  /// _______________  Data members  _______________________________________________

  /// _______________  Construction control  _______________________________________________

  RowSplineInfo* mConstructionRowSplineInfos = nullptr;  //! (transient!!) Temporary container of the row infos during construction
  IrregularSpline2D3D* mConstructionScenarios = nullptr; //! (transient!!) Temporary container for spline scenarios

  /// _______________  Geometry  _______________________________________________

  TPCFastTransformGeo mGeo; ///< TPC geometry information

  int mNumberOfScenarios; ///< Number of approximation spline scenarios

  RowSplineInfo* mRowSplineInfoPtr;  //! (transient!!) pointer to RowInfo array inside the mFlatBufferPtr buffer
  IrregularSpline2D3D* mScenarioPtr; //! (transient!!) pointer to spline scenarios

  /// _______________  Calibration data  _______________________________________________

  long int mTimeStamp; ///< time stamp of the current calibration

  char* mSplineData;          //! (transient!!) pointer to the spline data in the flat buffer
  size_t mSliceDataSizeBytes; ///< size of the data for one slice in the flat buffer
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

GPUdi() int TPCDistortionIRS::getDistortion(int slice, int row, float u, float v, float& dx, float& du, float& dv) const
{
  const IrregularSpline2D3D& spline = getSpline(slice, row);
  const float* splineData = getSplineData(slice, row);
  float su = 0, sv = 0;
  mGeo.convUVtoScaledUV(slice, row, u, v, su, sv);
  spline.getSplineVec(splineData, su, sv, dx, du, dv);
  return 0;
}

GPUdi() const IrregularSpline2D3D& TPCDistortionIRS::getSpline(int slice, int row) const
{
  /// Gives pointer to spline
  const RowSplineInfo& rowInfo = mRowSplineInfoPtr[row];
  return mScenarioPtr[rowInfo.splineScenarioID];
}

GPUdi() float* TPCDistortionIRS::getSplineDataNonConst(int slice, int row)
{
  /// Gives pointer to spline data
  const RowSplineInfo& rowInfo = mRowSplineInfoPtr[row];
  return reinterpret_cast<float*>(mSplineData + mSliceDataSizeBytes * slice + rowInfo.dataOffsetBytes);
}

GPUdi() const float* TPCDistortionIRS::getSplineData(int slice, int row) const
{
  /// Gives pointer to spline data
  const RowSplineInfo& rowInfo = mRowSplineInfoPtr[row];
  return reinterpret_cast<float*>(mSplineData + mSliceDataSizeBytes * slice + rowInfo.dataOffsetBytes);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
