// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCDistortionIRS.cxx
/// \brief Implementation of TPCDistortionIRS class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "TPCDistortionIRS.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

using namespace GPUCA_NAMESPACE::gpu;

TPCDistortionIRS::TPCDistortionIRS()
  : FlatObject(), mConstructionCounterRows(0), mConstructionCounterScenarios(0), mConstructionRowInfos(nullptr), mConstructionScenarios(nullptr), mNumberOfRows(0), mNumberOfScenarios(0), mRowInfoPtr(nullptr), mScenarioPtr(nullptr), mScaleVtoSVsideA(0.f), mScaleVtoSVsideC(0.f), mScaleSVtoVsideA(0.f), mScaleSVtoVsideC(0.f), mTimeStamp(-1), mSplineData(nullptr), mSliceDataSizeBytes(0)
{
  // Default Constructor: creates an empty uninitialized object
}

void TPCDistortionIRS::destroy()
{
  mConstructionCounterRows = 0;
  mConstructionCounterScenarios = 0;
  mConstructionRowInfos.reset();
  mConstructionScenarios.reset();
  mNumberOfRows = 0;
  mNumberOfScenarios = 0;
  mRowInfoPtr = nullptr;
  mScenarioPtr = nullptr;
  mScaleVtoSVsideA = 0.f;
  mScaleVtoSVsideC = 0.f;
  mScaleSVtoVsideA = 0.f;
  mScaleSVtoVsideC = 0.f;
  mTimeStamp = -1;
  mSplineData = nullptr;
  mSliceDataSizeBytes = 0;
  FlatObject::destroy();
}

void TPCDistortionIRS::relocateBufferPointers(const char* oldBuffer, char* newBuffer)
{
  mRowInfoPtr = FlatObject::relocatePointer(oldBuffer, newBuffer, mRowInfoPtr);
  mScenarioPtr = FlatObject::relocatePointer(oldBuffer, newBuffer, mScenarioPtr);

  for (int i = 0; i < mNumberOfScenarios; i++) {
    IrregularSpline2D3D& sp = mScenarioPtr[i];
    char* newSplineBuf = relocatePointer(oldBuffer, newBuffer, sp.getFlatBufferPtr());
    sp.setActualBufferAddress(newSplineBuf);
  }
  mSplineData = relocatePointer(oldBuffer, newBuffer, mSplineData);
}

void TPCDistortionIRS::cloneFromObject(const TPCDistortionIRS& obj, char* newFlatBufferPtr)
{
  /// Initializes from another object, copies data to newBufferPtr
  /// When newBufferPtr==nullptr, an internal container will be created, the data will be copied there.
  /// If there are any pointers inside the buffer, they has to be relocated (currently no pointers).

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;

  FlatObject::cloneFromObject(obj, newFlatBufferPtr);

  mConstructionCounterRows = 0;
  mConstructionCounterScenarios = 0;
  mConstructionRowInfos.reset();
  mConstructionScenarios.reset();

  mNumberOfRows = obj.mNumberOfRows;
  mNumberOfScenarios = obj.mNumberOfScenarios;

  mScaleVtoSVsideA = obj.mScaleVtoSVsideA;
  mScaleVtoSVsideC = obj.mScaleVtoSVsideC;
  mScaleSVtoVsideA = obj.mScaleSVtoVsideA;
  mScaleSVtoVsideC = obj.mScaleSVtoVsideC;

  mTimeStamp = obj.mTimeStamp;

  mSliceDataSizeBytes = obj.mSliceDataSizeBytes;

  // variable-size data
  mRowInfoPtr = obj.mRowInfoPtr;
  mScenarioPtr = obj.mScenarioPtr;
  mSplineData = obj.mSplineData;

  relocateBufferPointers(oldFlatBufferPtr, mFlatBufferPtr);
}

void TPCDistortionIRS::moveBufferTo(char* newFlatBufferPtr)
{
  /// Sets buffer pointer to the new address, move the buffer content there.

  const char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  relocateBufferPointers(oldFlatBufferPtr, mFlatBufferPtr);
}

void TPCDistortionIRS::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// Sets the actual location of the external flat buffer after it has been moved (i.e. to another maschine)
  const char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  relocateBufferPointers(oldFlatBufferPtr, mFlatBufferPtr);
}

void TPCDistortionIRS::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// Sets a future location of the external flat buffer before moving it to this location (i.e. when copying to GPU).
  /// The object can be used immidiatelly after the move, call of setActualFlatBufferAddress() is not needed.
  /// !!! Information about the actual buffer location will be lost.
  /// !!! Most of the class methods may be called only after the buffer will be moved to its new location.
  /// !!! To undo call setActualFlatBufferAddress()
  ///

  char* oldBuffer = mFlatBufferPtr;
  char* newBuffer = futureFlatBufferPtr;

  mRowInfoPtr = relocatePointer(oldBuffer, newBuffer, mRowInfoPtr);

  for (int i = 0; i < mNumberOfScenarios; i++) {
    IrregularSpline2D3D& sp = mScenarioPtr[i];
    char* newSplineBuf = relocatePointer(oldBuffer, newBuffer, sp.getFlatBufferPtr());
    sp.setFutureBufferAddress(newSplineBuf);
  }
  mScenarioPtr = relocatePointer(oldBuffer, newBuffer, mScenarioPtr);
  mSplineData = relocatePointer(oldBuffer, newBuffer, mSplineData);

  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

void TPCDistortionIRS::startConstruction(int numberOfRows, int numberOfScenarios)
{
  /// Starts the construction procedure, reserves temporary memory

  FlatObject::startConstruction();

  assert((numberOfRows > 0) && (numberOfScenarios > 0));

  mNumberOfRows = numberOfRows;
  mNumberOfScenarios = numberOfScenarios;

  mConstructionCounterRows = 0;
  mConstructionCounterScenarios = 0;
  mConstructionRowInfos.reset(new RowInfo[numberOfRows]);
  mConstructionScenarios.reset(new IrregularSpline2D3D[numberOfScenarios]);

  mTimeStamp = -1;

  mRowInfoPtr = nullptr;
  mScenarioPtr = nullptr;
  mScaleVtoSVsideA = 0.f;
  mScaleVtoSVsideC = 0.f;
  mScaleSVtoVsideA = 0.f;
  mScaleSVtoVsideC = 0.f;
  mSplineData = nullptr;
  mSliceDataSizeBytes = 0;
}

void TPCDistortionIRS::setTPCrow(int iRow, float x, int nPads, float padWidth, int iScenario)
{
  /// Initializes a TPC row
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(iRow >= 0 && iRow < mNumberOfRows && nPads > 1 && iScenario >= 0 && iScenario < mNumberOfScenarios);

  RowInfo& row = mConstructionRowInfos[iRow];

  double uWidth = (nPads - 1) * padWidth;
  row.x = x;
  row.U0 = -uWidth / 2;
  row.scaleUtoSU = 1. / uWidth;
  row.scaleSUtoU = uWidth;
  row.splineScenarioID = iScenario;
  row.dataOffsetBytes = 0;
  mConstructionCounterRows++;
}

void TPCDistortionIRS::setTPCgeometry(float tpcLengthSideA, float tpcLengthSideC)
{
  /// Sets TPC geometry
  /// It must be called once during construction
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(tpcLengthSideA > 1.f && tpcLengthSideC > 1.f);
  mScaleVtoSVsideA = 1. / tpcLengthSideA;
  mScaleVtoSVsideC = 1. / tpcLengthSideC;
  mScaleSVtoVsideA = tpcLengthSideA;
  mScaleSVtoVsideC = tpcLengthSideC;
  mConstructionMask |= ConstructionExtraState::GeometryIsSet;
}

void TPCDistortionIRS::setApproximationScenario(int scenarioIndex, const IrregularSpline2D3D& spline)
{
  /// Sets approximation scenario
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(scenarioIndex >= 0 && scenarioIndex < mNumberOfScenarios);
  assert(spline.isConstructed());
  IrregularSpline2D3D& sp = mConstructionScenarios[scenarioIndex];
  sp.cloneFromObject(spline, nullptr); //  clone to internal buffer container
  mConstructionCounterScenarios++;
}

void TPCDistortionIRS::finishConstruction()
{
  /// Finishes construction: puts everything to the flat buffer, releases temporary memory

  assert(mConstructionMask & ConstructionState::InProgress);
  assert(mConstructionMask & ConstructionExtraState::GeometryIsSet);
  assert(mConstructionCounterRows == mNumberOfRows);
  assert(mConstructionCounterScenarios == mNumberOfScenarios);

  // organize memory for the flat buffer and caculate its size

  size_t rowsOffset = 0;
  size_t rowsSize = sizeof(RowInfo) * mNumberOfRows;

  size_t scOffset = alignSize(rowsOffset + rowsSize, IrregularSpline2D3D::getClassAlignmentBytes());
  size_t scSize = sizeof(IrregularSpline2D3D) * mNumberOfScenarios;

  size_t scBufferOffsets[mNumberOfScenarios];

  scBufferOffsets[0] = alignSize(scOffset + scSize, IrregularSpline2D3D::getBufferAlignmentBytes());
  size_t scBufferSize = 0;
  for (int i = 0; i < mNumberOfScenarios; i++) {
    IrregularSpline2D3D& sp = mConstructionScenarios[i];
    scBufferOffsets[i] = scBufferOffsets[0] + scBufferSize;
    scBufferSize = alignSize(scBufferSize + sp.getFlatBufferSize(), sp.getBufferAlignmentBytes());
  }

  size_t sliceDataOffset = alignSize(scBufferOffsets[0] + scBufferSize, IrregularSpline2D3D::getDataAlignmentBytes());

  mSliceDataSizeBytes = 0;
  for (int i = 0; i < mNumberOfRows; i++) {
    RowInfo& row = mConstructionRowInfos[i];
    row.dataOffsetBytes = mSliceDataSizeBytes;
    IrregularSpline2D3D& sp = mConstructionScenarios[row.splineScenarioID];
    mSliceDataSizeBytes += 3 * sp.getNumberOfKnots() * sizeof(float);
    mSliceDataSizeBytes = alignSize(mSliceDataSizeBytes, IrregularSpline2D3D::getDataAlignmentBytes());
  }

  FlatObject::finishConstruction(sliceDataOffset + mSliceDataSizeBytes * NumberOfSlices);

  mRowInfoPtr = reinterpret_cast<RowInfo*>(mFlatBufferPtr + rowsOffset);
  for (int i = 0; i < mNumberOfRows; i++) {
    mRowInfoPtr[i] = mConstructionRowInfos[i];
  }

  mScenarioPtr = reinterpret_cast<IrregularSpline2D3D*>(mFlatBufferPtr + scOffset);

  for (int i = 0; i < mNumberOfScenarios; i++) {
    IrregularSpline2D3D& sp0 = mConstructionScenarios[i];
    IrregularSpline2D3D& sp1 = mScenarioPtr[i];
    new (&sp1) IrregularSpline2D3D(); // first, call a constructor
    sp1.cloneFromObject(sp0, mFlatBufferPtr + scBufferOffsets[i]);
  }

  mSplineData = reinterpret_cast<char*>(mFlatBufferPtr + sliceDataOffset);

  mConstructionCounterRows = 0;
  mConstructionCounterScenarios = 0;
  mConstructionRowInfos.reset();
  mConstructionScenarios.reset();

  mTimeStamp = -1;

  // initialise all distortions to 0.

  for (int slice = 0; slice < NumberOfSlices; slice++) {
    for (int row = 0; row < mNumberOfRows; row++) {
      const IrregularSpline2D3D& spline = getSpline(slice, row);
      float* data = getSplineDataNonConst(slice, row);
      for (int i = 0; i < 3 * spline.getNumberOfKnots(); i++) {
        data[i] = 0.f;
      }
      spline.correctEdges(data);
    }
  }
}

void TPCDistortionIRS::Print() const
{
#if !defined(GPUCA_GPUCODE)
  std::cout << " TPC DistortionIRS: " << std::endl;
  std::cout << "  mNumberOfRows = " << mNumberOfRows << std::endl;
  std::cout << "  mNumberOfScenarios = " << mNumberOfScenarios << std::endl;
  std::cout << "  mScaleVtoSVsideA = " << mScaleVtoSVsideA << std::endl;
  std::cout << "  mScaleVtoSVsideC = " << mScaleVtoSVsideC << std::endl;
  std::cout << "  mScaleSVtoVsideA = " << mScaleSVtoVsideA << std::endl;
  std::cout << "  mScaleSVtoVsideC = " << mScaleSVtoVsideC << std::endl;
  std::cout << "  mTimeStamp = " << mTimeStamp << std::endl;
  std::cout << "  mSliceDataSizeBytes = " << mSliceDataSizeBytes << std::endl;
  std::cout << "  TPC rows: " << std::endl;
  for (int i = 0; i < mNumberOfRows; i++) {
    RowInfo& r = mRowInfoPtr[i];
    std::cout << " tpc row " << i << ": x = " << r.x << " U0 = " << r.U0 << " scaleUtoSU = " << r.scaleUtoSU << " scaleSUtoU = " << r.scaleSUtoU << " splineScenarioID = " << r.splineScenarioID << " dataOffsetBytes = " << r.dataOffsetBytes << std::endl;
  }
  for (int i = 0; i < mNumberOfScenarios; i++) {
    std::cout << " SplineScenario " << i << ": " << std::endl;
    mScenarioPtr[i].Print();
  }
  std::cout << " Spline Data: " << std::endl;
  for (int is = 0; is < NumberOfSlices; is++) {
    for (int ir = 0; ir < mNumberOfRows; ir++) {
      std::cout << "slice " << is << " row " << ir << ": " << std::endl;
      const IrregularSpline2D3D& spline = getSpline(is, ir);
      const float* d = getSplineData(is, ir);
      int k = 0;
      for (int i = 0; i < spline.getGridU().getNumberOfKnots(); i++) {
        for (int j = 0; j < spline.getGridV().getNumberOfKnots(); j++, k++) {
          std::cout << d[k] << " ";
        }
        std::cout << std::endl;
      }
    }
  }
#endif
}
