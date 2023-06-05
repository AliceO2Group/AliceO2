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

/// \file  TPCFastSpaceChargeCorrection.cxx
/// \brief Implementation of TPCFastSpaceChargeCorrection class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "TPCFastSpaceChargeCorrection.h"
#include "GPUCommonLogger.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#include <cmath>
#include "ChebyshevFit1D.h"
#include "Spline2DHelper.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

TPCFastSpaceChargeCorrection::TPCFastSpaceChargeCorrection()
  : FlatObject(),
    mConstructionRowInfos(nullptr),
    mConstructionScenarios(nullptr),
    mNumberOfScenarios(0),
    mScenarioPtr(nullptr),
    mRowInfoPtr(nullptr),
    mSliceRowInfoPtr(nullptr),
    mTimeStamp(-1),
    mSplineData{nullptr, nullptr, nullptr},
    mSliceDataSizeBytes{0, 0, 0}
{
  // Default Constructor: creates an empty uninitialized object
}

TPCFastSpaceChargeCorrection::~TPCFastSpaceChargeCorrection()
{
  /// Destructor
  destroy();
}

void TPCFastSpaceChargeCorrection::releaseConstructionMemory()
{
// release temporary arrays
#if !defined(GPUCA_GPUCODE)
  delete[] mConstructionRowInfos;
  delete[] mConstructionScenarios;
#endif
  mConstructionRowInfos = nullptr;
  mConstructionScenarios = nullptr;
}

void TPCFastSpaceChargeCorrection::destroy()
{
  releaseConstructionMemory();
  mConstructionRowInfos = nullptr;
  mConstructionScenarios = nullptr;
  mNumberOfScenarios = 0;
  mRowInfoPtr = nullptr;
  mSliceRowInfoPtr = nullptr;
  mScenarioPtr = nullptr;
  mTimeStamp = -1;
  for (int is = 0; is < 3; is++) {
    mSplineData[is] = nullptr;
    mSliceDataSizeBytes[is] = 0;
  }
  FlatObject::destroy();
}

void TPCFastSpaceChargeCorrection::relocateBufferPointers(const char* oldBuffer, char* newBuffer)
{
  mRowInfoPtr = FlatObject::relocatePointer(oldBuffer, newBuffer, mRowInfoPtr);
  mSliceRowInfoPtr = FlatObject::relocatePointer(oldBuffer, newBuffer, mSliceRowInfoPtr);
  mScenarioPtr = FlatObject::relocatePointer(oldBuffer, newBuffer, mScenarioPtr);

  for (int i = 0; i < mNumberOfScenarios; i++) {
    SplineType& sp = mScenarioPtr[i];
    char* newSplineBuf = relocatePointer(oldBuffer, newBuffer, sp.getFlatBufferPtr());
    sp.setActualBufferAddress(newSplineBuf);
  }
  mSplineData[0] = relocatePointer(oldBuffer, newBuffer, mSplineData[0]);
  mSplineData[1] = relocatePointer(oldBuffer, newBuffer, mSplineData[1]);
  mSplineData[2] = relocatePointer(oldBuffer, newBuffer, mSplineData[2]);
}

void TPCFastSpaceChargeCorrection::cloneFromObject(const TPCFastSpaceChargeCorrection& obj, char* newFlatBufferPtr)
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

  for (int i = 0; i < TPCFastTransformGeo::getNumberOfSlices(); ++i) {
    mSliceInfo[i] = obj.mSliceInfo[i];
  }

  mSliceDataSizeBytes[0] = obj.mSliceDataSizeBytes[0];
  mSliceDataSizeBytes[1] = obj.mSliceDataSizeBytes[1];
  mSliceDataSizeBytes[2] = obj.mSliceDataSizeBytes[2];

  // variable-size data
  mRowInfoPtr = obj.mRowInfoPtr;
  mSliceRowInfoPtr = obj.mSliceRowInfoPtr;
  mScenarioPtr = obj.mScenarioPtr;
  mSplineData[0] = obj.mSplineData[0];
  mSplineData[1] = obj.mSplineData[1];
  mSplineData[2] = obj.mSplineData[2];

  relocateBufferPointers(oldFlatBufferPtr, mFlatBufferPtr);
}

void TPCFastSpaceChargeCorrection::moveBufferTo(char* newFlatBufferPtr)
{
  /// Sets buffer pointer to the new address, move the buffer content there.

  const char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  relocateBufferPointers(oldFlatBufferPtr, mFlatBufferPtr);
}

void TPCFastSpaceChargeCorrection::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// Sets the actual location of the external flat buffer after it has been moved (i.e. to another maschine)

  FlatObject::setActualBufferAddress(actualFlatBufferPtr);

  size_t rowsOffset = 0;
  size_t rowsSize = sizeof(RowInfo) * mGeo.getNumberOfRows();

  mRowInfoPtr = reinterpret_cast<RowInfo*>(mFlatBufferPtr + rowsOffset);

  size_t sliceRowsOffset = rowsOffset + rowsSize;
  size_t sliceRowsSize = sizeof(SliceRowInfo) * mGeo.getNumberOfRows() * mGeo.getNumberOfSlices();

  mSliceRowInfoPtr = reinterpret_cast<SliceRowInfo*>(mFlatBufferPtr + sliceRowsOffset);

  size_t scOffset = alignSize(sliceRowsOffset + sliceRowsSize, SplineType::getClassAlignmentBytes());
  size_t scSize = sizeof(SplineType) * mNumberOfScenarios;

  mScenarioPtr = reinterpret_cast<SplineType*>(mFlatBufferPtr + scOffset);

  size_t scBufferOffset = alignSize(scOffset + scSize, SplineType::getBufferAlignmentBytes());
  size_t scBufferSize = 0;

  for (int i = 0; i < mNumberOfScenarios; i++) {
    SplineType& sp = mScenarioPtr[i];
    sp.setActualBufferAddress(mFlatBufferPtr + scBufferOffset + scBufferSize);
    scBufferSize = alignSize(scBufferSize + sp.getFlatBufferSize(), sp.getBufferAlignmentBytes());
  }
  size_t bufferSize = scBufferOffset + scBufferSize;
  for (int is = 0; is < 3; is++) {
    size_t sliceDataOffset = alignSize(bufferSize, SplineType::getParameterAlignmentBytes());
    mSplineData[is] = reinterpret_cast<char*>(mFlatBufferPtr + sliceDataOffset);
    bufferSize = sliceDataOffset + mSliceDataSizeBytes[is] * mGeo.getNumberOfSlices();
  }
}

void TPCFastSpaceChargeCorrection::setFutureBufferAddress(char* futureFlatBufferPtr)
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
  mSliceRowInfoPtr = relocatePointer(oldBuffer, newBuffer, mSliceRowInfoPtr);

  for (int i = 0; i < mNumberOfScenarios; i++) {
    SplineType& sp = mScenarioPtr[i];
    char* newSplineBuf = relocatePointer(oldBuffer, newBuffer, sp.getFlatBufferPtr());
    sp.setFutureBufferAddress(newSplineBuf);
  }
  mScenarioPtr = relocatePointer(oldBuffer, newBuffer, mScenarioPtr);
  mSplineData[0] = relocatePointer(oldBuffer, newBuffer, mSplineData[0]);
  mSplineData[1] = relocatePointer(oldBuffer, newBuffer, mSplineData[1]);
  mSplineData[2] = relocatePointer(oldBuffer, newBuffer, mSplineData[2]);

  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

void TPCFastSpaceChargeCorrection::print() const
{
  LOG(info) << " TPC Correction: ";
  mGeo.print();
  LOG(info) << "  mNumberOfScenarios = " << mNumberOfScenarios;
  LOG(info) << "  mTimeStamp = " << mTimeStamp;
  LOG(info) << "  mSliceDataSizeBytes = " << mSliceDataSizeBytes[0] << " " << mSliceDataSizeBytes[1] << " " << mSliceDataSizeBytes[2];
  if (mRowInfoPtr) {
    LOG(info) << "  TPC rows: ";
    for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
      RowInfo& r = mRowInfoPtr[i];
      LOG(info) << " tpc row " << i << ": splineScenarioID = " << r.splineScenarioID << " dataOffsetBytes = " << r.dataOffsetBytes;
    }
  }
  if (mScenarioPtr) {
    for (int i = 0; i < mNumberOfScenarios; i++) {
      LOG(info) << " SplineScenario " << i << ": ";
      mScenarioPtr[i].print();
    }
  }
  if (mRowInfoPtr && mScenarioPtr && mSliceRowInfoPtr) {
    LOG(info) << " Spline Data: ";
    for (int is = 0; is < mGeo.getNumberOfSlices(); is++) {
      for (int ir = 0; ir < mGeo.getNumberOfRows(); ir++) {
        LOG(info) << "slice " << is << " row " << ir << ": ";
        const SplineType& spline = getSpline(is, ir);
        const float* d = getSplineData(is, ir);
        int k = 0;
        for (int i = 0; i < spline.getGridX1().getNumberOfKnots(); i++) {
          for (int j = 0; j < spline.getGridX2().getNumberOfKnots(); j++, k++) {
            LOG(info) << d[k] << " ";
          }
          LOG(info) << "";
        }
      }
      //    LOG(info) << "inverse correction: slice " << slice
      //            << " dx " << maxDslice[0] << " du " << maxDslice[1] << " dv " << maxDslice[2] ;
    }
  }
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

void TPCFastSpaceChargeCorrection::startConstruction(const TPCFastTransformGeo& geo, int numberOfSplineScenarios)
{
  /// Starts the construction procedure, reserves temporary memory

  FlatObject::startConstruction();

  assert((geo.isConstructed()) && (numberOfSplineScenarios > 0));

  mGeo = geo;
  mNumberOfScenarios = numberOfSplineScenarios;

  releaseConstructionMemory();

#if !defined(GPUCA_GPUCODE)
  mConstructionRowInfos = new RowInfo[mGeo.getNumberOfRows()];
  mConstructionScenarios = new SplineType[mNumberOfScenarios];
#endif

  assert(mConstructionRowInfos != nullptr);
  assert(mConstructionScenarios != nullptr);

  for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
    mConstructionRowInfos[i].splineScenarioID = -1;
  }

  for (int i = 0; i < mNumberOfScenarios; i++) {
    mConstructionScenarios[i].destroy();
  }

  mTimeStamp = -1;

  mRowInfoPtr = nullptr;
  mSliceRowInfoPtr = nullptr;
  mScenarioPtr = nullptr;
  for (int s = 0; s < 3; s++) {
    mSplineData[s] = nullptr;
    mSliceDataSizeBytes[s] = 0;
  }
}

void TPCFastSpaceChargeCorrection::setRowScenarioID(int iRow, int iScenario)
{
  /// Initializes a TPC row
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(iRow >= 0 && iRow < mGeo.getNumberOfRows() && iScenario >= 0 && iScenario < mNumberOfScenarios);

  RowInfo& row = mConstructionRowInfos[iRow];
  row.splineScenarioID = iScenario;
  for (int s = 0; s < 3; s++) {
    row.dataOffsetBytes[s] = 0;
  }
}

void TPCFastSpaceChargeCorrection::setSplineScenario(int scenarioIndex, const SplineType& spline)
{
  /// Sets approximation scenario
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(scenarioIndex >= 0 && scenarioIndex < mNumberOfScenarios);
  assert(spline.isConstructed());
  SplineType& sp = mConstructionScenarios[scenarioIndex];
  sp.cloneFromObject(spline, nullptr); //  clone to internal buffer container
}

void TPCFastSpaceChargeCorrection::finishConstruction()
{
  /// Finishes construction: puts everything to the flat buffer, releases temporary memory

  assert(mConstructionMask & ConstructionState::InProgress);

  for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
    assert(mConstructionRowInfos[i].splineScenarioID >= 0);
  }
  for (int i = 0; i < mNumberOfScenarios; i++) {
    assert(mConstructionScenarios[i].isConstructed());
  }

  // organize memory for the flat buffer and caculate its size

  size_t rowsOffset = 0;
  size_t rowsSize = sizeof(RowInfo) * mGeo.getNumberOfRows();

  size_t sliceRowsOffset = rowsSize;
  size_t sliceRowsSize = sizeof(SliceRowInfo) * mGeo.getNumberOfRows() * mGeo.getNumberOfSlices();

  size_t scOffset = alignSize(sliceRowsOffset + sliceRowsSize, SplineType::getClassAlignmentBytes());
  size_t scSize = sizeof(SplineType) * mNumberOfScenarios;

  size_t scBufferOffsets[mNumberOfScenarios];

  scBufferOffsets[0] = alignSize(scOffset + scSize, SplineType::getBufferAlignmentBytes());
  size_t scBufferSize = 0;
  for (int i = 0; i < mNumberOfScenarios; i++) {
    SplineType& sp = mConstructionScenarios[i];
    scBufferOffsets[i] = scBufferOffsets[0] + scBufferSize;
    scBufferSize = alignSize(scBufferSize + sp.getFlatBufferSize(), sp.getBufferAlignmentBytes());
  }
  size_t bufferSize = scBufferOffsets[0] + scBufferSize;
  size_t sliceDataOffset[3];
  for (int is = 0; is < 3; is++) {
    sliceDataOffset[is] = alignSize(bufferSize, SplineType::getParameterAlignmentBytes());
    mSliceDataSizeBytes[is] = 0;
    for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
      RowInfo& row = mConstructionRowInfos[i];
      SplineType& spline = mConstructionScenarios[row.splineScenarioID];
      row.dataOffsetBytes[is] = alignSize(mSliceDataSizeBytes[is], SplineType::getParameterAlignmentBytes());
      mSliceDataSizeBytes[is] = row.dataOffsetBytes[is] + spline.getSizeOfParameters();
    }
    mSliceDataSizeBytes[is] = alignSize(mSliceDataSizeBytes[is], SplineType::getParameterAlignmentBytes());
    bufferSize = sliceDataOffset[is] + mSliceDataSizeBytes[is] * mGeo.getNumberOfSlices();
  }

  FlatObject::finishConstruction(bufferSize);

  mRowInfoPtr = reinterpret_cast<RowInfo*>(mFlatBufferPtr + rowsOffset);
  for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
    mRowInfoPtr[i] = mConstructionRowInfos[i];
  }

  mSliceRowInfoPtr = reinterpret_cast<SliceRowInfo*>(mFlatBufferPtr + sliceRowsOffset);
  for (int s = 0; s < mGeo.getNumberOfSlices(); s++) {
    for (int r = 0; r < mGeo.getNumberOfRows(); r++) {
      mSliceRowInfoPtr[s * mGeo.getNumberOfRows() + r].gridCorrU0 = 0.;
      mSliceRowInfoPtr[s * mGeo.getNumberOfRows() + r].scaleCorrUtoGrid = 0.;
      mSliceRowInfoPtr[s * mGeo.getNumberOfRows() + r].scaleCorrVtoGrid = 0.;
    }
  }

  mScenarioPtr = reinterpret_cast<SplineType*>(mFlatBufferPtr + scOffset);

  for (int i = 0; i < mNumberOfScenarios; i++) {
    SplineType& sp0 = mConstructionScenarios[i];
    SplineType& sp1 = mScenarioPtr[i];
    new (&sp1) SplineType(); // first, call a constructor
    sp1.cloneFromObject(sp0, mFlatBufferPtr + scBufferOffsets[i]);
  }

  for (int is = 0; is < 3; is++) {
    mSplineData[is] = reinterpret_cast<char*>(mFlatBufferPtr + sliceDataOffset[is]);
  }
  releaseConstructionMemory();

  mTimeStamp = -1;

  setNoCorrection();
}

GPUd() void TPCFastSpaceChargeCorrection::setNoCorrection()
{
  // initialise all corrections to 0.
  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    double vLength = (slice < mGeo.getNumberOfSlicesA()) ? mGeo.getTPCzLengthA() : mGeo.getTPCzLengthC();
    SliceInfo& sliceInfo = getSliceInfo(slice);
    sliceInfo.vMax = vLength;
    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      const SplineType& spline = getSpline(slice, row);

      for (int is = 0; is < 3; is++) {
        float* data = getSplineData(slice, row, is);
        int nPar = spline.getNumberOfParameters();
        if (is == 1) {
          nPar = nPar / 3;
        }
        if (is == 2) {
          nPar = nPar * 2 / 3;
        }
        for (int i = 0; i < nPar; i++) {
          data[i] = 0.f;
        }
      }

      SliceRowInfo& info = getSliceRowInfo(slice, row);
      RowActiveArea& area = info.activeArea;
      for (int i = 1; i < 5; i++) {
        area.maxDriftLengthCheb[i] = 0;
      }
      area.maxDriftLengthCheb[0] = vLength;
      area.cuMin = mGeo.convPadToU(row, 0.f);
      area.cuMax = -area.cuMin;
      area.vMax = vLength;
      area.cvMax = vLength;
      info.gridCorrU0 = area.cuMin;
      info.scaleCorrUtoGrid = (spline.getGridX1().getNumberOfKnots() - 1) / (area.cuMax - area.cuMin);
      info.scaleCorrVtoGrid = (spline.getGridX2().getNumberOfKnots() - 1) / area.cvMax;
    } // row
  }   // slice
}

void TPCFastSpaceChargeCorrection::constructWithNoCorrection(const TPCFastTransformGeo& geo)
{
  const int nCorrectionScenarios = 1;
  startConstruction(geo, nCorrectionScenarios);
  for (int row = 0; row < geo.getNumberOfRows(); row++) {
    setRowScenarioID(row, 0);
  }
  {
    TPCFastSpaceChargeCorrection::SplineType spline;
    spline.recreate(2, 2);
    setSplineScenario(0, spline);
  }
  finishConstruction();
  setNoCorrection();
  initInverse();
}

void TPCFastSpaceChargeCorrection::initMaxDriftLength(bool prn)
{
  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSlicesA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  ChebyshevFit1D chebFitter;

  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    if (prn) {
      LOG(info) << "init MaxDriftLength for slice " << slice;
    }
    double vLength = (slice < mGeo.getNumberOfSlicesA()) ? mGeo.getTPCzLengthA() : mGeo.getTPCzLengthC();
    SliceInfo& sliceInfo = getSliceInfo(slice);
    sliceInfo.vMax = 0.f;

    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      RowActiveArea& area = getSliceRowInfo(slice, row).activeArea;
      area.cvMax = 0;
      area.vMax = 0;
      area.cuMin = mGeo.convPadToU(row, 0.f);
      area.cuMax = -area.cuMin;
      chebFitter.reset(4, 0., mGeo.getRowInfo(row).maxPad);
      double x = mGeo.getRowInfo(row).x;
      for (int pad = 0; pad < mGeo.getRowInfo(row).maxPad; pad++) {
        float u = mGeo.convPadToU(row, (float)pad);
        float v0 = 0;
        float v1 = 1.1 * vLength;
        float vLastValid = -1;
        float cvLastValid = -1;
        while (v1 - v0 > 0.1) {
          float v = 0.5 * (v0 + v1);
          float dx, du, dv;
          getCorrection(slice, row, u, v, dx, du, dv);
          double cx = x + dx;
          double cu = u + du;
          double cv = v + dv;
          double r2 = cx * cx + cu * cu;
          if (cv < 0) {
            v0 = v;
          } else if (cv <= vLength && r2 >= tpcR2min && r2 <= tpcR2max) {
            v0 = v;
            vLastValid = v;
            cvLastValid = cv;
          } else {
            v1 = v;
          }
        }
        if (vLastValid > 0.) {
          chebFitter.addMeasurement(pad, vLastValid);
        }
        if (area.vMax < vLastValid) {
          area.vMax = vLastValid;
        }
        if (area.cvMax < cvLastValid) {
          area.cvMax = cvLastValid;
        }
      }
      chebFitter.fit();
      for (int i = 0; i < 5; i++) {
        area.maxDriftLengthCheb[i] = chebFitter.getCoefficients()[i];
      }
      if (sliceInfo.vMax < area.vMax) {
        sliceInfo.vMax = area.vMax;
      }
    } // row
  }   // slice
}

void TPCFastSpaceChargeCorrection::initInverse(bool prn)
{

  initMaxDriftLength(prn);

  Spline2DHelper<float> helper;
  std::vector<float> splineParameters;

  ChebyshevFit1D chebFitterX, chebFitterU, chebFitterV;

  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSlicesA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    // LOG(info) << "inverse transform for slice " << slice ;
    double vLength = (slice < mGeo.getNumberOfSlicesA()) ? mGeo.getTPCzLengthA() : mGeo.getTPCzLengthC();
    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      SplineType spline = getSpline(slice, row);
      helper.setSpline(spline, 30, 30);
      std::vector<double> dataPointCU, dataPointCV, dataPointF;

      float u0, u1, v0, v1;
      mGeo.convScaledUVtoUV(slice, row, 0., 0., u0, v0);
      mGeo.convScaledUVtoUV(slice, row, 1., 1., u1, v1);

      double x = mGeo.getRowInfo(row).x;
      int nPointsU = (spline.getGridX1().getNumberOfKnots() - 1) * 30;
      int nPointsV = (spline.getGridX2().getNumberOfKnots() - 1) * 30;

      double stepU = (u1 - u0) / (nPointsU - 1);
      double stepV = (v1 - v0) / (nPointsV - 1);

      if (prn) {
        LOG(info) << "u0 " << u0 << " u1 " << u1 << " v0 " << v0 << " v1 " << v1;
      }
      RowActiveArea& area = getSliceRowInfo(slice, row).activeArea;
      area.cuMin = 1.e10;
      area.cuMax = -1.e10;

      /*
      v1 = area.vMax;
      stepV = (v1 - v0) / (nPointsU - 1);
      if (stepV < 1.f) {
        stepV = 1.f;
      }
      */

      for (double u = u0; u < u1 + stepU; u += stepU) {
        for (double v = v0; v < v1 + stepV; v += stepV) {
          float dx, du, dv;
          getCorrection(slice, row, u, v, dx, du, dv);
          double cx = x + dx;
          double cu = u + du;
          double cv = v + dv;
          if (cu < area.cuMin) {
            area.cuMin = cu;
          }
          if (cu > area.cuMax) {
            area.cuMax = cu;
          }

          dataPointCU.push_back(cu);
          dataPointCV.push_back(cv);
          dataPointF.push_back(dx);
          dataPointF.push_back(du);
          dataPointF.push_back(dv);

          if (prn) {
            LOG(info) << "measurement cu " << cu << " cv " << cv << " dx " << dx << " du " << du << " dv " << dv;
          }
        } // v
      }   // u

      if (area.cuMax - area.cuMin < 0.2) {
        area.cuMax = .1;
        area.cuMin = -.1;
      }
      if (area.cvMax < 0.1) {
        area.cvMax = .1;
      }
      if (prn) {
        LOG(info) << "slice " << slice << " row " << row << " max drift L = " << getMaxDriftLength(slice, row)
                  << " active area: cuMin " << area.cuMin << " cuMax " << area.cuMax << " vMax " << area.vMax << " cvMax " << area.cvMax;
      }

      SliceRowInfo& info = mSliceRowInfoPtr[slice * mGeo.getNumberOfRows() + row];
      info.gridCorrU0 = area.cuMin;
      info.scaleCorrUtoGrid = spline.getGridX1().getUmax() / (area.cuMax - area.cuMin);
      info.scaleCorrVtoGrid = spline.getGridX2().getUmax() / area.cvMax;

      info.gridCorrU0 = u0;
      info.gridCorrV0 = info.gridV0;
      info.scaleCorrUtoGrid = spline.getGridX1().getUmax() / (u1 - info.gridCorrU0);
      info.scaleCorrVtoGrid = spline.getGridX2().getUmax() / (v1 - info.gridCorrV0);

      int nDataPoints = dataPointCU.size();
      for (int i = 0; i < nDataPoints; i++) {
        dataPointCU[i] = (dataPointCU[i] - info.gridCorrU0) * info.scaleCorrUtoGrid;
        dataPointCV[i] = (dataPointCV[i] - info.gridCorrV0) * info.scaleCorrVtoGrid;
      }

      splineParameters.resize(spline.getNumberOfParameters());

      helper.approximateDataPoints(spline, splineParameters.data(), 0., spline.getGridX1().getUmax(),
                                   0., spline.getGridX2().getUmax(),
                                   dataPointCU.data(), dataPointCV.data(),
                                   dataPointF.data(), dataPointCU.size());

      float* splineX = getSplineData(slice, row, 1);
      float* splineUV = getSplineData(slice, row, 2);
      for (int i = 0; i < spline.getNumberOfParameters() / 3; i++) {
        splineX[i] = splineParameters[3 * i + 0];
        splineUV[2 * i + 0] = splineParameters[3 * i + 1];
        splineUV[2 * i + 1] = splineParameters[3 * i + 2];
      }
    } // row
  }   // slice
}

#ifdef XXX
void TPCFastSpaceChargeCorrection::initInverse(bool prn)
{

  initMaxDriftLength(prn);

  Spline2DHelper<float> helper;
  std::vector<double> dataPointF;
  std::vector<float> splineParameters;

  ChebyshevFit1D chebFitterX, chebFitterU, chebFitterV;

  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSlicesA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    // LOG(info) << "inverse transform for slice " << slice ;
    double vLength = (slice < mGeo.getNumberOfSlicesA()) ? mGeo.getTPCzLengthA() : mGeo.getTPCzLengthC();
    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      const SplineType& spline = getSpline(slice, row);
      helper.setSpline(spline, 3, 3);

      float u0, u1, v0, v1;
      mGeo.convScaledUVtoUV(slice, row, 0., 0., u0, v0);
      mGeo.convScaledUVtoUV(slice, row, 1., 1., u1, v1);

      double x = mGeo.getRowInfo(row).x;
      double stepU = (u1 - u0) / (1. * (helper.getNumberOfDataPointsU1() - 1));
      double stepV = (v1 - v0) / (1. * (helper.getNumberOfDataPointsU2() - 1));

      if (prn) {
        LOG(info) << "u0 " << u0 << " u1 " << u1 << " v0 " << v0 << " v1 " << v1;
      }
      RowActiveArea& area = getSliceRowInfo(slice, row).activeArea;
      area.cuMin = 1.e10;
      area.cuMax = -1.e10;

      v1 = area.vMax;
      stepV = (v1 - v0) / (1. * (helper.getNumberOfDataPointsU2() - 1));
      if (stepV < 1.f) {
        stepV = 1.f;
      }
      int nCheb = helper.getNumberOfDataPointsU2();
      nCheb = 20;
      chebFitterV.reset(nCheb - 1, 0, vLength);

      struct Entry {
        double cu, cv, dx, du, dv;
      };
      std::vector<Entry> dataRowsV[helper.getNumberOfDataPointsU2()];

      for (double u = u0; u < u1 + stepU; u += stepU) {
        chebFitterV.reset();
        //double vvMax = 0;
        for (double v = v0; v < v1 + stepV; v += stepV) {
          float dx, du, dv;
          getCorrectionOld(slice, row, u, v, dx, du, dv);
          double cx = x + dx;
          double cu = u + du;
          double cv = v + dv;
          double r2 = cx * cx + cu * cu;
          if (cv < 0 || cv > vLength || r2 < tpcR2min || r2 > tpcR2max) {
            // SG!!continue;
          }
          if (cu < area.cuMin) {
            area.cuMin = cu;
          }
          if (cu > area.cuMax) {
            area.cuMax = cu;
          }
          //if (v > vvMax) {
          //vvMax = v;
          //}
          if (prn) {
            LOG(info) << "measurement cu " << cu << " cv " << cv << " dx " << dx << " du " << du << " dv " << dv;
          }
          chebFitterV.addMeasurement(cv, dv);
        } // v
        if (prn) {
          LOG(info) << "u " << u << " nmeas " << chebFitterV.getNmeasurements();
        }
        if (chebFitterV.getNmeasurements() < 1) {
          continue;
        }
        chebFitterV.fit();
        if (prn) {
          LOG(info) << "slice " << slice << " row " << row << " u " << u;
          LOG(info) << "n cheb " << nCheb << " n measurements " << chebFitterV.getNmeasurements();
          for (int i = 0; i < nCheb; i++) {
            LOG(info) << i << " " << chebFitterV.getCoefficients()[i] << " ";
          }
          LOG(info);
          //exit(0);
        }
        // TODO: refit with extra measurements close to cv == data points cv

        // fill data for cv data rows
        double drow = area.cvMax / (helper.getNumberOfDataPointsU2() - 1);
        for (int i = 0; i < helper.getNumberOfDataPointsU2(); i++) {
          double cv = i * drow;
          double dvCheb = chebFitterV.eval(cv);
          double v = cv - dvCheb;
          // weighted combination between cheb and nominal
          //if (v < 0 || v > vvMax) {
          //continue;
          //}

          float dx, du, dv;
          getCorrectionOld(slice, row, u, v, dx, du, dv);
          // LOG(info)<<" u "<<u<<" cv0 "<<cv<<" v "<<v<<" cu "<<u+du<<" cv "<<v+dv;
          double cu = u + du;
          cv = v + dv;
          // double cx = x + dx;
          // double r2 = cx * cx + cu * cu;

          //if (cv < 0 || cv > vLength || r2 < tpcR2min || r2 > tpcR2max) {
          //continue;
          //}
          Entry e{cu, cv, dx, du, dv};
          dataRowsV[i].push_back(e);
        }
      } // u

      if (prn) {
        /// maximal possible drift length of the active area
        LOG(info) << " cuMin " << area.cuMin << " cuMax " << area.cuMax << " cvMax " << area.cvMax;
      }
      if (area.cuMax - area.cuMin < 0.2) {
        area.cuMax = .1;
        area.cuMin = -.1;
      }
      if (area.cvMax < 0.1) {
        area.cvMax = .1;
      }
      SliceRowInfo& info = mSliceRowInfoPtr[slice * mGeo.getNumberOfRows() + row];
      info.gridCorrU0 = area.cuMin;
      info.scaleCorrUtoGrid = (spline.getGridX1().getNumberOfKnots() - 1) / (area.cuMax - area.cuMin);
      info.scaleCorrVtoGrid = (spline.getGridX2().getNumberOfKnots() - 1) / area.cvMax;

      dataPointF.resize(helper.getNumberOfDataPoints() * 3);

      // fit u(cu)
      nCheb = helper.getNumberOfDataPointsU1();
      nCheb = 20;
      chebFitterX.reset(nCheb - 1, area.cuMin, area.cuMax);
      chebFitterU.reset(nCheb - 1, area.cuMin, area.cuMax);
      chebFitterV.reset(nCheb - 1, area.cuMin, area.cuMax);

      // double drow = area.cvMax / (helper.getNumberOfDataPointsU2() - 1);
      double dcol = (area.cuMax - area.cuMin) / (helper.getNumberOfDataPointsU1() - 1);
      for (int iv = 0; iv < helper.getNumberOfDataPointsU2(); iv++) {
        // double cv = iv * drow;
        double* dataPointFrow = &dataPointF[iv * helper.getNumberOfDataPointsU1() * 3];
        for (int iu = 0; iu < helper.getNumberOfDataPointsU1(); iu++) {
          dataPointFrow[iu * 3 + 0] = 0;
          dataPointFrow[iu * 3 + 1] = 0;
          dataPointFrow[iu * 3 + 2] = 0;
        }
        chebFitterX.reset();
        chebFitterU.reset();
        chebFitterV.reset();
        for (unsigned int i = 0; i < dataRowsV[iv].size(); i++) {
          chebFitterX.addMeasurement(dataRowsV[iv][i].cu, dataRowsV[iv][i].dx);
          chebFitterU.addMeasurement(dataRowsV[iv][i].cu, dataRowsV[iv][i].du);
          chebFitterV.addMeasurement(dataRowsV[iv][i].cu, dataRowsV[iv][i].dv);
        }
        if (chebFitterU.getNmeasurements() < 1) {
          continue;
        }

        chebFitterX.fit();
        chebFitterU.fit();
        chebFitterV.fit();

        // fill data points
        for (int iu = 0; iu < helper.getNumberOfDataPointsU1(); iu++) {
          double cu = area.cuMin + iu * dcol;
          dataPointFrow[iu * 3 + 0] = chebFitterX.eval(cu);
          dataPointFrow[iu * 3 + 1] = chebFitterU.eval(cu);
          dataPointFrow[iu * 3 + 2] = chebFitterV.eval(cu);
        } // iu
      }   // iv

      splineParameters.resize(spline.getNumberOfParameters());
      helper.approximateFunction(splineParameters.data(), dataPointF.data());
      float* splineX = getSplineData(slice, row, 1);
      float* splineUV = getSplineData(slice, row, 2);
      for (int i = 0; i < spline.getNumberOfParameters() / 3; i++) {
        splineX[i] = splineParameters[3 * i + 0];
        splineUV[2 * i + 0] = splineParameters[3 * i + 1];
        splineUV[2 * i + 1] = splineParameters[3 * i + 2];
      }
    } // row
  }   // slice
}
#endif

double TPCFastSpaceChargeCorrection::testInverse(bool prn)
{
  if (prn) {
    LOG(info) << "Test inverse transform ";
  }

  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSlicesA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  double maxDtpc[3] = {0, 0, 0};
  double maxD = 0;

  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    if (prn) {
      LOG(info) << "check inverse transform for slice " << slice;
    }
    double vLength = (slice < mGeo.getNumberOfSlicesA()) ? mGeo.getTPCzLengthA() : mGeo.getTPCzLengthC();
    double maxDslice[3] = {0, 0, 0};
    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      float u0, u1, v0, v1;
      mGeo.convScaledUVtoUV(slice, row, 0., 0., u0, v0);
      mGeo.convScaledUVtoUV(slice, row, 1., 1., u1, v1);
      double x = mGeo.getRowInfo(row).x;
      double stepU = (u1 - u0) / 100.;
      double stepV = (v1 - v0) / 100.;
      double maxDrow[3] = {0, 0, 0};
      for (double u = u0; u < u1; u += stepU) {
        for (double v = v0; v < v1; v += stepV) {
          float dx, du, dv;
          getCorrection(slice, row, u, v, dx, du, dv);
          double cx = x + dx;
          double cu = u + du;
          double cv = v + dv;
          double r2 = cx * cx + cu * cu;
          if (cv < 0 || cv > vLength || r2 < tpcR2min || r2 > tpcR2max) {
            continue;
          }
          float nx, nu, nv;
          getCorrectionInvCorrectedX(slice, row, cu, cv, nx);
          getCorrectionInvUV(slice, row, cu, cv, nu, nv);
          double d[3] = {nx - cx, nu - u, nv - v};
          for (int i = 0; i < 3; i++) {
            if (fabs(d[i]) > fabs(maxDrow[i])) {
              maxDrow[i] = d[i];
            }
          }

          if (0 && prn && fabs(d[0]) + fabs(d[1]) + fabs(d[2]) > 0.1) {
            LOG(info) << nx - cx << " " << nu - u << " " << nv - v
                      << " x,u,v " << x << ", " << u << ", " << v
                      << " dx,du,dv " << cx - x << ", " << cu - u << ", " << cv - v
                      << " nx,nu,nv " << nx - x << ", " << cu - nu << ", " << cv - nv;
          }
        }
      }
      if (0 && prn) {
        LOG(info) << "slice " << slice << " row " << row
                  << " dx " << maxDrow[0] << " du " << maxDrow[1] << " dv " << maxDrow[2];
      }
      for (int i = 0; i < 3; i++) {
        if (fabs(maxDslice[i]) < fabs(maxDrow[i])) {
          maxDslice[i] = maxDrow[i];
        }
        if (fabs(maxDtpc[i]) < fabs(maxDrow[i])) {
          maxDtpc[i] = maxDrow[i];
        }
        if (fabs(maxD) < fabs(maxDrow[i])) {
          maxD = maxDrow[i];
        }
      }
    }
    if (prn) {
      LOG(info) << "inverse correction: slice " << slice
                << " dx " << maxDslice[0] << " du " << maxDslice[1] << " dv " << maxDslice[2];
    }
  } // slice

  LOG(info) << "Test inverse TPC correction. max deviations: "
            << " dx " << maxDtpc[0] << " du " << maxDtpc[1] << " dv " << maxDtpc[2] << " cm";

  return maxD;
}

#endif // GPUCA_GPUCODE
