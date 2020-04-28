// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCFastSpaceChargeCorrection.cxx
/// \brief Implementation of TPCFastSpaceChargeCorrection class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "TPCFastSpaceChargeCorrection.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#include <cmath>
#include "ChebyshevFit1D.h"
#include "SplineHelper2D.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

TPCFastSpaceChargeCorrection::TPCFastSpaceChargeCorrection()
  : FlatObject(),
    mConstructionRowInfos(nullptr),
    mConstructionScenarios(nullptr),
    mNumberOfScenarios(0),
    mRowInfoPtr(nullptr),
    mSliceRowInfoPtr(nullptr),
    mScenarioPtr(nullptr),
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
      mSliceRowInfoPtr[s * mGeo.getNumberOfRows() + r].CorrU0 = 0;
      mSliceRowInfoPtr[s * mGeo.getNumberOfRows() + r].scaleCorrUtoGrid = 0;
      mSliceRowInfoPtr[s * mGeo.getNumberOfRows() + r].scaleCorrVtoGrid = 0;
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

  // initialise all corrections to 0.
  for (int is = 0; is < 3; is++) {
    for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
      for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
        const SplineType& spline = getSpline(slice, row);
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
    }
  }
}

#if !defined(GPUCA_GPUCODE)

void TPCFastSpaceChargeCorrection::print() const
{
  std::cout << " TPC Correction: " << std::endl;
  mGeo.print();
  std::cout << "  mNumberOfScenarios = " << mNumberOfScenarios << std::endl;
  std::cout << "  mTimeStamp = " << mTimeStamp << std::endl;
  std::cout << "  mSliceDataSizeBytes = " << mSliceDataSizeBytes << std::endl;
  std::cout << "  TPC rows: " << std::endl;
  for (int i = 0; i < mGeo.getNumberOfRows(); i++) {
    RowInfo& r = mRowInfoPtr[i];
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
      const SplineType& spline = getSpline(is, ir);
      const float* d = getSplineData(is, ir);
      int k = 0;
      for (int i = 0; i < spline.getGridU1().getNumberOfKnots(); i++) {
        for (int j = 0; j < spline.getGridU2().getNumberOfKnots(); j++, k++) {
          std::cout << d[k] << " ";
        }
        std::cout << std::endl;
      }
    }
  }
}

GPUh() void TPCFastSpaceChargeCorrection::initInverse()
{

  SplineHelper2D<float> helper;
  std::vector<float> dataPointF;
  std::vector<float> splineParameters;

  ChebyshevFit1D chebFitterU, chebFitterV;

  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSlicesA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    std::cout << "inverse transform for slice " << slice << std::endl;
    //if (slice != 0)
    //continue;
    double vLength = (slice < mGeo.getNumberOfSlicesA()) ? mGeo.getTPCzLengthA() : mGeo.getTPCzLengthC();
    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      //if (row != 0)
      //continue;
      float cuMin = 1.e10, cuMax = -1.e10, cvMax = 0.;
      const SplineType& spline = getSpline(slice, row);
      helper.setSpline(spline, 2, 2);

      float u0, u1, v0, v1;
      mGeo.convScaledUVtoUV(slice, row, 0., 0., u0, v0);
      mGeo.convScaledUVtoUV(slice, row, 1., 1., u1, v1);
      double x = mGeo.getRowInfo(row).x;
      double stepU = (u1 - u0) / (1. * (helper.getNumberOfDataPointsU1() - 1));
      double stepV = (v1 - v0) / (1. * (helper.getNumberOfDataPointsU2() - 1));

      //std::cout << "u0 " << u0 << " u1 " << u1 << " v0 " << v0 << " v1 " << v1 << std::endl;
      int nCheb = helper.getNumberOfDataPointsU2();
      nCheb = 20;
      chebFitterV.reset(nCheb - 1, 0, vLength);

      struct Entry {
        double cu, cv, du, dv;
      };
      std::vector<Entry> dataRowsV[helper.getNumberOfDataPointsU2()];

      for (double u = u0; u <= u1; u += stepU) {
        chebFitterV.reset();
        double vMax = 0;
        for (double v = v0; v <= v1; v += stepV) {
          float dx, du, dv;
          getCorrection(slice, row, u, v, dx, du, dv);
          double cx = x + dx;
          double cu = u + du;
          double cv = v + dv;
          double r2 = cx * cx + cu * cu;
          if (cv < 0 || cv > vLength || r2 < tpcR2min || r2 > tpcR2max) {
            continue;
          }
          if (cu < cuMin) {
            cuMin = cu;
          }
          if (cu > cuMax) {
            cuMax = cu;
          }
          if (cv > cvMax) {
            cvMax = cv;
          }
          if (v > vMax) {
            vMax = v;
          }
          //std::cout << cv / vLength << " " << v / vLength << std::endl;
          chebFitterV.addMeasurement(cv, dv);
        } // v
        //std::cout << "u " << u << " nmeas " << chebFitterV.getNmeasurements() << std::endl;
        if (chebFitterV.getNmeasurements() < 1) {
          continue;
        }
        chebFitterV.fit();
        if (0) {
          std::cout << "slice " << slice << " row " << row << std::endl;
          std::cout << "n cheb " << nCheb << " n measurements " << chebFitterV.getNmeasurements()
                    << std::endl;
          for (int i = 0; i < nCheb; i++) {
            std::cout << i << " " << chebFitterV.getCoefficients()[i] << std::endl;
          }
          exit(0);
        }
        // TODO: refit with extra measurements close to cv == data points cv

        // fill data for cv data rows
        double drow = vLength / (helper.getNumberOfDataPointsU2() - 1);
        for (int i = 0; i < helper.getNumberOfDataPointsU2(); i++) {
          double cv = i * drow;
          double dvCheb = chebFitterV.eval(cv);
          double v = cv - dvCheb;
          /* //SG weighted combination between cheb and nominal
          if (v < 0 || v > vMax) {
            continue;
          }
          */
          float dx, du, dv;
          getCorrection(slice, row, u, v, dx, du, dv);
          //std::cout<<" u "<<u<<" cv0 "<<cv<<" v "<<v<<" cu "<<u+du<<" cv "<<v+dv<<std::endl;
          double cu = u + du;
          cv = v + dv;
          double cx = x + dx;
          double r2 = cx * cx + cu * cu;
          /*
          if (cv < 0 || cv > vLength || r2 < tpcR2min || r2 > tpcR2max) {
            continue;
          }*/
          Entry e{cu, cv, du, dv};
          //std::cout<<"m 1, row V "<<i<<std::endl;
          dataRowsV[i].push_back(e);
        }
      } // u

      //cuMin = 0; //SG!!!
      //std::cout << " cuMin " << cuMin << " cuMax " << cuMax << " cvMax " << cvMax << std::endl;

      SliceRowInfo& info = mSliceRowInfoPtr[slice * mGeo.getNumberOfRows() + row];
      info.CorrU0 = cuMin;
      info.scaleCorrUtoGrid = (spline.getGridU1().getNumberOfKnots() - 1) / (cuMax - cuMin);
      info.scaleCorrVtoGrid = (spline.getGridU2().getNumberOfKnots() - 1) / vLength;

      dataPointF.resize(helper.getNumberOfDataPoints() * 3);

      // fit u(cu)
      nCheb = helper.getNumberOfDataPointsU1();
      nCheb = 20;
      chebFitterU.reset(nCheb - 1, cuMin, cuMax);
      chebFitterV.reset(nCheb - 1, cuMin, cuMax);

      double drow = vLength / (helper.getNumberOfDataPointsU2() - 1);
      double dcol = (cuMax - cuMin) / (helper.getNumberOfDataPointsU1() - 1);
      for (int iv = 0; iv < helper.getNumberOfDataPointsU2(); iv++) {
        double cv = iv * drow;
        float* dataPointFrow = &dataPointF[iv * helper.getNumberOfDataPointsU1() * 3];
        for (int iu = 0; iu < helper.getNumberOfDataPointsU1(); iu++) {
          //double cu = cuMin + iu * dcol;
          //double cv = iv * drow;
          dataPointFrow[iu * 3 + 0] = 0; //x;
          dataPointFrow[iu * 3 + 1] = 0; //cu;
          dataPointFrow[iu * 3 + 2] = 0; //cv;
        }                                // iu

        chebFitterU.reset();
        chebFitterV.reset();
        for (unsigned int i = 0; i < dataRowsV[iv].size(); i++) {
          chebFitterU.addMeasurement(dataRowsV[iv][i].cu, dataRowsV[iv][i].du);
          chebFitterV.addMeasurement(dataRowsV[iv][i].cu, dataRowsV[iv][i].dv);
        }
        if (chebFitterU.getNmeasurements() < 1) {
          continue;
        }

        chebFitterU.fit();
        chebFitterV.fit();

        // fill data points
        for (int iu = 0; iu < helper.getNumberOfDataPointsU1(); iu++) {
          double cu = cuMin + iu * dcol;
          double du0 = chebFitterU.eval(cu);
          double dv0 = chebFitterV.eval(cu);
          double u = cu - du0;
          double v = cv - dv0;
          float dx, du, dv;
          getCorrection(slice, row, u, v, dx, du, dv);
          dataPointFrow[iu * 3 + 0] = dx; //cx;
          dataPointFrow[iu * 3 + 1] = du0;
          dataPointFrow[iu * 3 + 2] = dv0;
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
  } // slice
}

GPUh() void TPCFastSpaceChargeCorrection::testInverse()
{
  initInverse();

  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSlicesA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  for (int slice = 0; slice < mGeo.getNumberOfSlices(); slice++) {
    //if (slice != 0)
    //continue;
    //std::cout << "check inverse transform for slice " << slice << std::endl;
    double vLength = (slice < mGeo.getNumberOfSlicesA()) ? mGeo.getTPCzLengthA() : mGeo.getTPCzLengthC();
    double maxDslice[3] = {0, 0, 0};
    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      //if (row != 0)
      //continue;
      const SplineType& spline = getSpline(slice, row);
      float u0, u1, v0, v1;
      mGeo.convScaledUVtoUV(slice, row, 0., 0, u0, v0);
      mGeo.convScaledUVtoUV(slice, row, 1., 1, u1, v1);
      double x = mGeo.getRowInfo(row).x;
      double stepU = (u1 - u0) / 100.;
      double stepV = (v1 - v0) / 100.;
      double maxDrow[3] = {0, 0, 0};
      for (double u = u0; u < u1; u += stepU) {
        for (double v = v0; v < v1; v += stepV) {
          float dx, du, dv;
          getCorrection(slice, row, u, v, dx, du, dv);
          //dv = 0.1*v;  //SG!!
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
            if (fabs(d[i]) > maxDrow[i])
              maxDrow[i] = fabs(d[i]);
          }
          /*
          if (fabs(d[0]) > 0.01) {
            std::cout << nx - cx << " " << nu - u << " " << nv - v
                      << " x,u,v " << x << ", " << u << ", " << v
                      << " dx,du,dv " << cx - x << ", " << cu - u << ", " << cv - v
                      << " nx,nu,nv " << nx - x << ", " << cu - nu << ", " << cv - nv << std::endl;
          }
          */
        }
      }
      /*
      std::cout << "slice " << slice << " row " << row
                << " dx " << maxDrow[0] << " du " << maxDrow[1] << " dv " << maxDrow[2] << std::endl;
*/
      for (int i = 0; i < 3; i++) {
        if (maxDrow[i] > maxDslice[i])
          maxDslice[i] = maxDrow[i];
      }
    }
    /*
    std::cout << "slice " << slice
              << " dx " << maxDslice[0] << " du " << maxDslice[1] << " dv " << maxDslice[2] << std::endl;
      */
  }
}

#endif // GPUCA_GPUCODE
