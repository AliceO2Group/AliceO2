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
#include <string>
#include <cmath>
#include "Spline2DHelper.h"
#endif

using namespace o2::gpu;

ClassImp(TPCFastSpaceChargeCorrection);

TPCFastSpaceChargeCorrection::TPCFastSpaceChargeCorrection()
  : FlatObject(),
    mConstructionScenarios(nullptr),
    mNumberOfScenarios(0),
    mScenarioPtr(nullptr),
    mTimeStamp(-1),
    mSplineData{nullptr, nullptr, nullptr},
    mSectorDataSizeBytes{0, 0, 0}
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
  delete[] mConstructionScenarios;
#endif
  mConstructionScenarios = nullptr;
}

void TPCFastSpaceChargeCorrection::destroy()
{
  releaseConstructionMemory();
  mConstructionScenarios = nullptr;
  mNumberOfScenarios = 0;
  mScenarioPtr = nullptr;
  mTimeStamp = -1;
  for (int32_t is = 0; is < 3; is++) {
    mSplineData[is] = nullptr;
    mSectorDataSizeBytes[is] = 0;
  }
  FlatObject::destroy();
}

void TPCFastSpaceChargeCorrection::relocateBufferPointers(const char* oldBuffer, char* newBuffer)
{
  mScenarioPtr = FlatObject::relocatePointer(oldBuffer, newBuffer, mScenarioPtr);

  for (int32_t i = 0; i < mNumberOfScenarios; i++) {
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

  for (int32_t i = 0; i < TPCFastTransformGeo::getNumberOfSectors(); ++i) {
    mSectorInfo[i] = obj.mSectorInfo[i];
  }

  mSectorDataSizeBytes[0] = obj.mSectorDataSizeBytes[0];
  mSectorDataSizeBytes[1] = obj.mSectorDataSizeBytes[1];
  mSectorDataSizeBytes[2] = obj.mSectorDataSizeBytes[2];

  // variable-size data
  mScenarioPtr = obj.mScenarioPtr;
  mSplineData[0] = obj.mSplineData[0];
  mSplineData[1] = obj.mSplineData[1];
  mSplineData[2] = obj.mSplineData[2];

  mClassVersion = obj.mClassVersion;

  for (int32_t i = 0; i < TPCFastTransformGeo::getMaxNumberOfRows(); i++) {
    mRowInfos[i] = obj.mRowInfos[i];
  }

  for (int32_t i = 0; i < TPCFastTransformGeo::getNumberOfSectors() * TPCFastTransformGeo::getMaxNumberOfRows(); i++) {
    mSectorRowInfos[i] = obj.mSectorRowInfos[i];
  }

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
  /// Sets the actual location of the external flat buffer after it has been moved (e.g. to another maschine)

  if (mClassVersion != 4) {
    LOG(error) << "TPCFastSpaceChargeCorrection::setActualBufferAddress() called with class version " << mClassVersion << ". This is not supported.";
    return;
  }

  FlatObject::setActualBufferAddress(actualFlatBufferPtr);

  size_t scSize = sizeof(SplineType) * mNumberOfScenarios;

  mScenarioPtr = reinterpret_cast<SplineType*>(mFlatBufferPtr);

  size_t scBufferOffset = alignSize(scSize, SplineType::getBufferAlignmentBytes());
  size_t scBufferSize = 0;

  for (int32_t i = 0; i < mNumberOfScenarios; i++) {
    SplineType& sp = mScenarioPtr[i];
    sp.setActualBufferAddress(mFlatBufferPtr + scBufferOffset + scBufferSize);
    scBufferSize = alignSize(scBufferSize + sp.getFlatBufferSize(), sp.getBufferAlignmentBytes());
  }
  size_t bufferSize = scBufferOffset + scBufferSize;
  for (int32_t is = 0; is < 3; is++) {
    size_t sectorDataOffset = alignSize(bufferSize, SplineType::getParameterAlignmentBytes());
    mSplineData[is] = reinterpret_cast<char*>(mFlatBufferPtr + sectorDataOffset);
    bufferSize = sectorDataOffset + mSectorDataSizeBytes[is] * mGeo.getNumberOfSectors();
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

  for (int32_t i = 0; i < mNumberOfScenarios; i++) {
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
  LOG(info) << "  mSectorDataSizeBytes = " << mSectorDataSizeBytes[0] << " " << mSectorDataSizeBytes[1] << " " << mSectorDataSizeBytes[2];
  {
    LOG(info) << "  TPC rows: ";
    for (int32_t i = 0; i < mGeo.getNumberOfRows(); i++) {
      const RowInfo& r = mRowInfos[i];
      LOG(info) << " tpc row " << i << ": splineScenarioID = " << r.splineScenarioID << " dataOffsetBytes = " << r.dataOffsetBytes;
    }
  }
  if (mScenarioPtr) {
    for (int32_t i = 0; i < mNumberOfScenarios; i++) {
      LOG(info) << " SplineScenario " << i << ": ";
      mScenarioPtr[i].print();
    }
  }
  if (mScenarioPtr) {
    LOG(info) << " Spline Data: ";
    for (int32_t is = 0; is < mGeo.getNumberOfSectors(); is++) {
      for (int32_t ir = 0; ir < mGeo.getNumberOfRows(); ir++) {
        LOG(info) << "sector " << is << " row " << ir << ": ";
        const SplineType& spline = getSpline(is, ir);
        const float* d = getSplineData(is, ir);
        int32_t k = 0;
        for (int32_t i = 0; i < spline.getGridX1().getNumberOfKnots(); i++) {
          for (int32_t j = 0; j < spline.getGridX2().getNumberOfKnots(); j++, k++) {
            LOG(info) << d[k] << " ";
          }
          LOG(info) << "";
        }
      }
      //    LOG(info) << "inverse correction: sector " << sector
      //            << " dx " << maxDsector[0] << " du " << maxDsector[1] << " dv " << maxDsector[2] ;
    }
  }
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

void TPCFastSpaceChargeCorrection::startConstruction(const TPCFastTransformGeo& geo, int32_t numberOfSplineScenarios)
{
  /// Starts the construction procedure, reserves temporary memory

  FlatObject::startConstruction();

  assert((geo.isConstructed()) && (numberOfSplineScenarios > 0));

  mGeo = geo;
  mNumberOfScenarios = numberOfSplineScenarios;

  releaseConstructionMemory();

#if !defined(GPUCA_GPUCODE)
  mConstructionScenarios = new SplineType[mNumberOfScenarios];
#endif

  assert(mConstructionScenarios != nullptr);

  for (int32_t i = 0; i < mGeo.getNumberOfRows(); i++) {
    mRowInfos[i].splineScenarioID = -1;
  }

  for (int32_t i = 0; i < mNumberOfScenarios; i++) {
    mConstructionScenarios[i].destroy();
  }

  mTimeStamp = -1;

  mScenarioPtr = nullptr;
  for (int32_t s = 0; s < 3; s++) {
    mSplineData[s] = nullptr;
    mSectorDataSizeBytes[s] = 0;
  }
  mClassVersion = 4;
}

void TPCFastSpaceChargeCorrection::setRowScenarioID(int32_t iRow, int32_t iScenario)
{
  /// Initializes a TPC row
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(iRow >= 0 && iRow < mGeo.getNumberOfRows() && iScenario >= 0 && iScenario < mNumberOfScenarios);

  RowInfo& row = mRowInfos[iRow];
  row.splineScenarioID = iScenario;
  for (int32_t s = 0; s < 3; s++) {
    row.dataOffsetBytes[s] = 0;
  }
}

void TPCFastSpaceChargeCorrection::setSplineScenario(int32_t scenarioIndex, const SplineType& spline)
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

  for (int32_t i = 0; i < mGeo.getNumberOfRows(); i++) {
    assert(mRowInfos[i].splineScenarioID >= 0);
  }
  for (int32_t i = 0; i < mNumberOfScenarios; i++) {
    assert(mConstructionScenarios[i].isConstructed());
  }

  // organize memory for the flat buffer and caculate its size

  size_t scOffset = 0;
  size_t scSize = sizeof(SplineType) * mNumberOfScenarios;

  size_t scBufferOffsets[mNumberOfScenarios];

  scBufferOffsets[0] = alignSize(scOffset + scSize, SplineType::getBufferAlignmentBytes());
  size_t scBufferSize = 0;
  for (int32_t i = 0; i < mNumberOfScenarios; i++) {
    SplineType& sp = mConstructionScenarios[i];
    scBufferOffsets[i] = scBufferOffsets[0] + scBufferSize;
    scBufferSize = alignSize(scBufferSize + sp.getFlatBufferSize(), sp.getBufferAlignmentBytes());
  }
  size_t bufferSize = scBufferOffsets[0] + scBufferSize;
  size_t sectorDataOffset[3];
  for (int32_t is = 0; is < 3; is++) {
    sectorDataOffset[is] = alignSize(bufferSize, SplineType::getParameterAlignmentBytes());
    mSectorDataSizeBytes[is] = 0;
    for (int32_t i = 0; i < mGeo.getNumberOfRows(); i++) {
      RowInfo& row = mRowInfos[i];
      SplineType& spline = mConstructionScenarios[row.splineScenarioID];
      row.dataOffsetBytes[is] = alignSize(mSectorDataSizeBytes[is], SplineType::getParameterAlignmentBytes());
      mSectorDataSizeBytes[is] = row.dataOffsetBytes[is] + spline.getSizeOfParameters();
    }
    mSectorDataSizeBytes[is] = alignSize(mSectorDataSizeBytes[is], SplineType::getParameterAlignmentBytes());
    bufferSize = sectorDataOffset[is] + mSectorDataSizeBytes[is] * mGeo.getNumberOfSectors();
  }

  FlatObject::finishConstruction(bufferSize);

  mScenarioPtr = reinterpret_cast<SplineType*>(mFlatBufferPtr + scOffset);

  for (int32_t i = 0; i < mNumberOfScenarios; i++) {
    SplineType& sp0 = mConstructionScenarios[i];
    SplineType& sp1 = mScenarioPtr[i];
    new (&sp1) SplineType(); // first, call a constructor
    sp1.cloneFromObject(sp0, mFlatBufferPtr + scBufferOffsets[i]);
  }

  for (int32_t is = 0; is < 3; is++) {
    mSplineData[is] = reinterpret_cast<char*>(mFlatBufferPtr + sectorDataOffset[is]);
  }
  releaseConstructionMemory();

  mTimeStamp = -1;

  setNoCorrection();
}

GPUd() void TPCFastSpaceChargeCorrection::setNoCorrection()
{
  // initialise all corrections to 0.
  for (int32_t sector = 0; sector < mGeo.getNumberOfSectors(); sector++) {
    double vLength = mGeo.getTPCzLength();
    SectorInfo& sectorInfo = getSectorInfo(sector);
    sectorInfo.vMax = vLength;
    for (int32_t row = 0; row < mGeo.getNumberOfRows(); row++) {
      const SplineType& spline = getSpline(sector, row);

      for (int32_t is = 0; is < 3; is++) {
        float* data = getSplineData(sector, row, is);
        int32_t nPar = spline.getNumberOfParameters();
        if (is == 1) {
          nPar = nPar / 3;
        }
        if (is == 2) {
          nPar = nPar * 2 / 3;
        }
        for (int32_t i = 0; i < nPar; i++) {
          data[i] = 0.f;
        }
      }

      SectorRowInfo& info = getSectorRowInfo(sector, row);

      info.gridMeasured.y0 = mGeo.getRowInfo(row).getYmin();
      info.gridMeasured.yScale = spline.getGridX1().getUmax() / mGeo.getRowInfo(row).getYwidth();
      info.gridMeasured.l0 = 0.f;
      info.gridMeasured.lScale = spline.getGridX2().getUmax() / vLength;

      info.gridReal = info.gridMeasured;
    } // row
  } // sector
}

void TPCFastSpaceChargeCorrection::constructWithNoCorrection(const TPCFastTransformGeo& geo)
{
  const int32_t nCorrectionScenarios = 1;
  startConstruction(geo, nCorrectionScenarios);
  for (int32_t row = 0; row < geo.getNumberOfRows(); row++) {
    setRowScenarioID(row, 0);
  }
  {
    TPCFastSpaceChargeCorrection::SplineType spline;
    spline.recreate(2, 2);
    setSplineScenario(0, spline);
  }
  finishConstruction();
  setNoCorrection();
}

double TPCFastSpaceChargeCorrection::testInverse(bool prn)
{
  if (prn) {
    LOG(info) << "Test inverse transform ";
  }

  double tpcR2min = mGeo.getRowInfo(0).x - 1.;
  tpcR2min = tpcR2min * tpcR2min;
  double tpcR2max = mGeo.getRowInfo(mGeo.getNumberOfRows() - 1).x;
  tpcR2max = tpcR2max / cos(2 * M_PI / mGeo.getNumberOfSectorsA() / 2) + 1.;
  tpcR2max = tpcR2max * tpcR2max;

  struct MaxValue {
    double V{0.};
    int Sector{-1};
    int Row{-1};

    void update(double v, int sector, int row)
    {
      if (fabs(v) > fabs(V)) {
        V = v;
        Sector = sector;
        Row = row;
      }
    }
    void update(const MaxValue& other)
    {
      update(other.V, other.Sector, other.Row);
    }

    std::string toString()
    {
      std::stringstream ss;
      ss << V << "(" << Sector << "," << Row << ")";
      return ss.str();
    }
  };

  MaxValue maxDtpc[3];
  MaxValue maxD;

  for (int32_t sector = 0; sector < mGeo.getNumberOfSectors(); sector++) {
    if (prn) {
      LOG(info) << "check inverse transform for sector " << sector;
    }
    double vLength = mGeo.getTPCzLength();
    MaxValue maxDsector[3];
    for (int32_t row = 0; row < mGeo.getNumberOfRows(); row++) {
      double x = mGeo.getRowInfo(row).x;
      auto [y0, y1] = mGeo.getRowInfo(row).getYrange();
      auto [z0, z1] = mGeo.getZrange(sector);

      // grid borders
      if (sector < mGeo.getNumberOfSectorsA()) {
        z1 = vLength - getSectorRowInfo(sector, row).gridMeasured.l0;
      } else {
        z0 = getSectorRowInfo(sector, row).gridMeasured.l0 - vLength;
      }

      double stepY = (y1 - y0) / 100.;
      double stepZ = (z1 - z0) / 100.;
      MaxValue maxDrow[3];
      for (double y = y0; y < y1; y += stepY) {
        for (double z = z0; z < z1; z += stepZ) {
          auto [dx, dy, dz] = getCorrectionLocal(sector, row, y, z);
          double realX = x + dx;
          double realY = y + dy;
          double realZ = z + dz;
          if (!isLocalInsideGrid(sector, row, y, z) || !isLocalInsideGrid(sector, row, realY, realZ)) {
            continue;
          }
          double r2 = realX * realX + realY * realY;
          if (realY < y0 || realY > y1 ||
              realZ < z0 || realZ > z1 ||
              r2 < tpcR2min || r2 > tpcR2max) {
            continue;
          }
          float dxr = getCorrectionXatRealYZ(sector, row, realY, realZ);
          auto [dyr, dzr] = getCorrectionYZatRealYZ(sector, row, realY, realZ);
          double d[3] = {dxr - dx, dyr - dy, dzr - dz};
          for (int32_t i = 0; i < 3; i++) {
            maxDrow[i].update(d[i], sector, row);
          }

          if (0 && prn && fabs(d[0]) + fabs(d[1]) + fabs(d[2]) > 0.1) {
            LOG(info) << dxr - dx << " " << dyr - dy << " " << dzr - dz
                      << " measured xyz " << x << ", " << y << ", " << z
                      << " dx,dy,dz from measured point " << dx << ", " << dy << ", " << dz
                      << " dx,dy,dz from real point " << dxr << ", " << dyr << ", " << dzr;
          }
        }
      }
      if (0 && prn) {
        LOG(info) << "sector " << sector << " row " << row
                  << " dx " << maxDrow[0].V << " dy " << maxDrow[1].V << " dz " << maxDrow[2].V;
      }
      for (int32_t i = 0; i < 3; i++) {
        maxDsector[i].update(maxDrow[i]);
        maxDtpc[i].update(maxDrow[i]);
        maxD.update(maxDrow[i]);
      }
    }
    if (prn) {
      LOG(info) << "inverse correction: sector " << sector << ". Max deviations: "
                << " dx " << maxDsector[0].toString() << " dy " << maxDsector[1].toString() << " dz " << maxDsector[2].toString();
    }
  } // sector

  LOG(info) << "Test inverse TPC correction. max deviations: "
            << " dx " << maxDtpc[0].toString() << " dy " << maxDtpc[1].toString() << " dz " << maxDtpc[2].toString() << " cm";

  return maxD.V;
}

#endif // GPUCA_GPUCODE
