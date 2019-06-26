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
  : FlatObject(),
    mConstructionRowSplineInfos(nullptr),
    mConstructionScenarios(nullptr),
    mNumberOfScenarios(0),
    mRowSplineInfoPtr(nullptr),
    mScenarioPtr(nullptr),
    mTimeStamp(-1),
    mSplineData(nullptr),
    mSliceDataSizeBytes(0)
{
  // Default Constructor: creates an empty uninitialized object
}

TPCDistortionIRS::~TPCDistortionIRS()
{
  /// Destructor
  destroy();
}

void TPCDistortionIRS::releaseConstructionMemory()
{
  // release temporary arrays
#if !defined(GPUCA_GPUCODE)
  delete[] mConstructionRowSplineInfos;
  delete[] mConstructionScenarios;
#endif
  mConstructionRowSplineInfos = nullptr;
  mConstructionScenarios = nullptr;
}

void TPCDistortionIRS::destroy()
{
  releaseConstructionMemory();
  mConstructionRowSplineInfos = nullptr;
  mConstructionScenarios = nullptr;
  mNumberOfScenarios = 0;
  mRowSplineInfoPtr = nullptr;
  mScenarioPtr = nullptr;
  mTimeStamp = -1;
  mSplineData = nullptr;
  mSliceDataSizeBytes = 0;
  FlatObject::destroy();
}

void TPCDistortionIRS::relocateBufferPointers(const char* oldBuffer, char* newBuffer)
{
  mRowSplineInfoPtr = FlatObject::relocatePointer(oldBuffer, newBuffer, mRowSplineInfoPtr);
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

  releaseConstructionMemory();

  mNumberOfScenarios = obj.mNumberOfScenarios;

  mGeo = obj.mGeo;

  mTimeStamp = obj.mTimeStamp;

  mSliceDataSizeBytes = obj.mSliceDataSizeBytes;

  // variable-size data
  mRowSplineInfoPtr = obj.mRowSplineInfoPtr;
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

  mRowSplineInfoPtr = relocatePointer(oldBuffer, newBuffer, mRowSplineInfoPtr);

  for (int i = 0; i < mNumberOfScenarios; i++) {
    IrregularSpline2D3D& sp = mScenarioPtr[i];
    char* newSplineBuf = relocatePointer(oldBuffer, newBuffer, sp.getFlatBufferPtr());
    sp.setFutureBufferAddress(newSplineBuf);
  }
  mScenarioPtr = relocatePointer(oldBuffer, newBuffer, mScenarioPtr);
  mSplineData = relocatePointer(oldBuffer, newBuffer, mSplineData);

  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

void TPCDistortionIRS::startConstruction(const TPCFastTransformGeo& geo, int numberOfSplineScenarios)
{
  /// Starts the construction procedure, reserves temporary memory

  FlatObject::startConstruction();

  assert((geo.isConstructed()) && (numberOfSplineScenarios > 0));

  mGeo = geo;
  mNumberOfScenarios = numberOfSplineScenarios;

  releaseConstructionMemory();

#if !defined(GPUCA_GPUCODE)
  mConstructionRowSplineInfos = new RowSplineInfo[mGeo.getNumberOfRows()];
  mConstructionScenarios = new IrregularSpline2D3D[mNumberOfScenarios];
#endif

  assert(mConstructionRowSplineInfos != nullptr);
  assert(mConstructionScenarios != nullptr);

  for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
    mConstructionRowSplineInfos[i].splineScenarioID = -1;
  }

  for (int i = 0; i < mNumberOfScenarios; i++) {
    mConstructionScenarios[i].destroy();
  }

  mTimeStamp = -1;

  mRowSplineInfoPtr = nullptr;
  mScenarioPtr = nullptr;
  mSplineData = nullptr;
  mSliceDataSizeBytes = 0;
}

void TPCDistortionIRS::setRowScenarioID(int iRow, int iScenario)
{
  /// Initializes a TPC row
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(iRow >= 0 && iRow < mGeo.getNumberOfRows() && iScenario >= 0 && iScenario < mNumberOfScenarios);

  RowSplineInfo& row = mConstructionRowSplineInfos[iRow];
  row.splineScenarioID = iScenario;
  row.dataOffsetBytes = 0;
}

void TPCDistortionIRS::setSplineScenario(int scenarioIndex, const IrregularSpline2D3D& spline)
{
  /// Sets approximation scenario
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(scenarioIndex >= 0 && scenarioIndex < mNumberOfScenarios);
  assert(spline.isConstructed());
  IrregularSpline2D3D& sp = mConstructionScenarios[scenarioIndex];
  sp.cloneFromObject(spline, nullptr); //  clone to internal buffer container
}

void TPCDistortionIRS::finishConstruction()
{
  /// Finishes construction: puts everything to the flat buffer, releases temporary memory

  assert(mConstructionMask & ConstructionState::InProgress);

  for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
    assert(mConstructionRowSplineInfos[i].splineScenarioID >= 0);
  }
  for (int i = 0; i < mNumberOfScenarios; i++) {
    assert(mConstructionScenarios[i].isConstructed());
  }

  // organize memory for the flat buffer and caculate its size

  size_t rowsOffset = 0;
  size_t rowsSize = sizeof(RowSplineInfo) * mGeo.getNumberOfRows();

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
  for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
    RowSplineInfo& row = mConstructionRowSplineInfos[i];
    row.dataOffsetBytes = mSliceDataSizeBytes;
    IrregularSpline2D3D& sp = mConstructionScenarios[row.splineScenarioID];
    mSliceDataSizeBytes += 3 * sp.getNumberOfKnots() * sizeof(float);
    mSliceDataSizeBytes = alignSize(mSliceDataSizeBytes, IrregularSpline2D3D::getDataAlignmentBytes());
  }

  FlatObject::finishConstruction(sliceDataOffset + mSliceDataSizeBytes * mGeo.getNumberOfSlices());

  mRowSplineInfoPtr = reinterpret_cast<RowSplineInfo*>(mFlatBufferPtr + rowsOffset);
  for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
    mRowSplineInfoPtr[i] = mConstructionRowSplineInfos[i];
  }

  mScenarioPtr = reinterpret_cast<IrregularSpline2D3D*>(mFlatBufferPtr + scOffset);

  for (int i = 0; i < mNumberOfScenarios; i++) {
    IrregularSpline2D3D& sp0 = mConstructionScenarios[i];
    IrregularSpline2D3D& sp1 = mScenarioPtr[i];
    new (&sp1) IrregularSpline2D3D(); // first, call a constructor
    sp1.cloneFromObject(sp0, mFlatBufferPtr + scBufferOffsets[i]);
  }

  mSplineData = reinterpret_cast<char*>(mFlatBufferPtr + sliceDataOffset);

  releaseConstructionMemory();

  mTimeStamp = -1;

  // initialise all distortions to 0.

  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      const IrregularSpline2D3D& spline = getSpline(slice, row);
      float* data = getSplineDataNonConst(slice, row);
      for (int i = 0; i < 3 * spline.getNumberOfKnots(); i++) {
        data[i] = 0.f;
      }
      spline.correctEdges(data);
    }
  }
}

void TPCDistortionIRS::print() const
{
#if !defined(GPUCA_GPUCODE)
  std::cout << " TPC DistortionIRS: " << std::endl;
  mGeo.print();
  std::cout << "  mNumberOfScenarios = " << mNumberOfScenarios << std::endl;
  std::cout << "  mTimeStamp = " << mTimeStamp << std::endl;
  std::cout << "  mSliceDataSizeBytes = " << mSliceDataSizeBytes << std::endl;
  std::cout << "  TPC rows: " << std::endl;
  for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
    RowSplineInfo& r = mRowSplineInfoPtr[i];
    std::cout << " tpc row " << i << ": splineScenarioID = " << r.splineScenarioID << " dataOffsetBytes = " << r.dataOffsetBytes << std::endl;
  }
  for (int i = 0; i < mNumberOfScenarios; i++) {
    std::cout << " SplineScenario " << i << ": " << std::endl;
    mScenarioPtr[i].print();
  }
  std::cout << " Spline Data: " << std::endl;
  for (int is = 0; is < mGeo.getNumberOfSlices(); is++) {
    for (int ir = 0; ir < mGeo.getNumberOfRows(); ir++) {
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
