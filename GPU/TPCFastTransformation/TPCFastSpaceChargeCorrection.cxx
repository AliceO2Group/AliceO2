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

#include <iostream>
#include <string>
#include <cmath>
#include <sstream>
#include "Spline2DHelper.h"

using namespace o2::gpu;

ClassImp(TPCFastSpaceChargeCorrection);

TPCFastSpaceChargeCorrection::TPCFastSpaceChargeCorrection()
  : FlatObject(),
    mConstructionScenarios(nullptr),
    mNumberOfScenarios(0),
    mScenarioPtr(nullptr),
    mTimeStamp(-1),
    mCorrectionData{nullptr, nullptr, nullptr},
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
  delete[] mConstructionScenarios;
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
    mCorrectionData[is] = nullptr;
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
  mCorrectionData[0] = relocatePointer(oldBuffer, newBuffer, mCorrectionData[0]);
  mCorrectionData[1] = relocatePointer(oldBuffer, newBuffer, mCorrectionData[1]);
  mCorrectionData[2] = relocatePointer(oldBuffer, newBuffer, mCorrectionData[2]);
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

  mSectorDataSizeBytes[0] = obj.mSectorDataSizeBytes[0];
  mSectorDataSizeBytes[1] = obj.mSectorDataSizeBytes[1];
  mSectorDataSizeBytes[2] = obj.mSectorDataSizeBytes[2];

  // variable-size data
  mScenarioPtr = obj.mScenarioPtr;
  mCorrectionData[0] = obj.mCorrectionData[0];
  mCorrectionData[1] = obj.mCorrectionData[1];
  mCorrectionData[2] = obj.mCorrectionData[2];

  mClassVersion = obj.mClassVersion;

  for (int32_t i = 0; i < TPCFastTransformGeo::getMaxNumberOfRows(); i++) {
    mRowInfos[i] = obj.mRowInfos[i];
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

  if (mClassVersion == 4) {
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
      bufferSize = alignSize(bufferSize, SplineType::getParameterAlignmentBytes());
      mCorrectionData[is] = reinterpret_cast<char*>(mFlatBufferPtr + bufferSize);
      bufferSize += mSectorDataSizeBytes[is] * mGeo.getNumberOfSectors();
    }
    return;
  }

  if (mClassVersion != 3) {
    LOG(fatal) << "TPCFastSpaceChargeCorrection::setActualBufferAddress() called with class version " << mClassVersion << ". This is not supported.";
    return;
  }

  // Class version 3

  struct RowInfoVersion3 {
    int32_t splineScenarioID{0};  ///< scenario index (which of Spline2D splines to use)
    size_t dataOffsetBytes[3]{0}; ///< offset for the spline data withing a TPC sector
  };

  struct RowActiveAreaVersion3 {
    float maxDriftLengthCheb[5]{0.f};
    float vMax{0.f};
    float cuMin{0.f};
    float cuMax{0.f};
    float cvMax{0.f};
  };

  struct SectorRowInfoVersion3 {
    float gridV0{0.f};           ///< V coordinate of the V-grid start
    float gridCorrU0{0.f};       ///< U coordinate of the U-grid start for corrected U
    float gridCorrV0{0.f};       ///< V coordinate of the V-grid start for corrected V
    float scaleCorrUtoGrid{0.f}; ///< scale corrected U to U-grid coordinate
    float scaleCorrVtoGrid{0.f}; ///< scale corrected V to V-grid coordinate
    RowActiveAreaVersion3 activeArea;
  };

  FlatObject::setActualBufferAddress(actualFlatBufferPtr);

  size_t oldRowsOffset = 0;
  size_t oldRowsSize = sizeof(RowInfoVersion3) * mGeo.getNumberOfRows();

  size_t oldSectorRowsOffset = oldRowsOffset + oldRowsSize;
  size_t oldSectorRowsSize = sizeof(SectorRowInfoVersion3) * mGeo.getNumberOfRows() * mGeo.getNumberOfSectors();

  size_t oldScenariosOffset = alignSize(oldSectorRowsOffset + oldSectorRowsSize, SplineType::getClassAlignmentBytes());
  size_t scenariosSize = sizeof(SplineType) * mNumberOfScenarios;

  SplineType* oldScenarioPtr = reinterpret_cast<SplineType*>(mFlatBufferPtr + oldScenariosOffset);

  { // copy old-format sector and row parameters from the buffer to the arrays

    auto* oldRowInfos = reinterpret_cast<RowInfoVersion3*>(mFlatBufferPtr + oldRowsOffset);
    auto* oldSectorRowInfos = reinterpret_cast<SectorRowInfoVersion3*>(mFlatBufferPtr + oldSectorRowsOffset);

    int32_t iSector = 0;

    for (int32_t iRow = 0; iRow < mGeo.getNumberOfRows(); iRow++) {
      RowInfoVersion3& oldRowInfo = oldRowInfos[iRow];
      SectorRowInfoVersion3& oldSectorRowInfo = oldSectorRowInfos[mGeo.getNumberOfRows() * iSector + iRow];

      // the spline buffer is not yet initialised, don't try to access knot positions etc
      const auto& spline = oldScenarioPtr[oldRowInfo.splineScenarioID];

      RowInfo& newRowInfo = getRowInfo(iRow);

      newRowInfo.splineScenarioID = oldRowInfo.splineScenarioID;
      for (int32_t is = 0; is < 3; is++) {
        newRowInfo.dataOffsetBytes[is] = oldRowInfo.dataOffsetBytes[is];
      }

      { // grid for the measured coordinates
        float y0 = mGeo.getRowInfo(iRow).yMin;
        float yScale = spline.getGridX1().getUmax() / mGeo.getRowInfo(iRow).getYwidth();
        float zOut = mGeo.getTPCzLength() - oldSectorRowInfo.gridV0;
        float z0 = -3.;
        float zScale = spline.getGridX2().getUmax() / (zOut - z0);
        newRowInfo.gridMeasured.set(y0, yScale, z0, zScale, zOut, mGeo.getTPCzLength());
      }
      newRowInfo.gridReal = newRowInfo.gridMeasured;
    }
  }

  // move spline scenarios to the new place in the buffer

  mScenarioPtr = reinterpret_cast<SplineType*>(mFlatBufferPtr);
  memmove((void*)mScenarioPtr, (const void*)oldScenarioPtr, scenariosSize);

  size_t oldScenariosBufferOffset = alignSize(oldScenariosOffset + scenariosSize, SplineType::getBufferAlignmentBytes());
  size_t scenariosBufferOffset = alignSize(scenariosSize, SplineType::getBufferAlignmentBytes());

  size_t oldScenariosBufferSize = 0;
  size_t scenariosBufferSize = 0;
  for (int32_t i = 0; i < mNumberOfScenarios; i++) {
    SplineType& sp = mScenarioPtr[i];
    char* oldAddress = mFlatBufferPtr + oldScenariosBufferOffset + oldScenariosBufferSize;
    char* newAddress = mFlatBufferPtr + scenariosBufferOffset + scenariosBufferSize;
    memmove(newAddress, oldAddress, sp.getFlatBufferSize());
    sp.setActualBufferAddress(newAddress);
    oldScenariosBufferSize = alignSize(oldScenariosBufferSize + sp.getFlatBufferSize(), sp.getBufferAlignmentBytes());
    scenariosBufferSize = alignSize(scenariosBufferSize + sp.getFlatBufferSize(), sp.getBufferAlignmentBytes());
  }

  size_t oldBufferSize = oldScenariosBufferOffset + oldScenariosBufferSize;
  size_t bufferSize = scenariosBufferOffset + scenariosBufferSize;

  // move spline data to the new place in the buffer

  for (int32_t is = 0; is < 3; is++) {
    size_t oldCorrectionDataOffset = alignSize(oldBufferSize, SplineType::getParameterAlignmentBytes());
    size_t correctionDataOffset = alignSize(bufferSize, SplineType::getParameterAlignmentBytes());
    mCorrectionData[is] = reinterpret_cast<char*>(mFlatBufferPtr + correctionDataOffset);
    memmove(mCorrectionData[is], mFlatBufferPtr + oldCorrectionDataOffset, mSectorDataSizeBytes[is] * mGeo.getNumberOfSectors());
    oldBufferSize = oldCorrectionDataOffset + mSectorDataSizeBytes[is] * mGeo.getNumberOfSectors();
    bufferSize = correctionDataOffset + mSectorDataSizeBytes[is] * mGeo.getNumberOfSectors();
  }

  mFlatBufferSize = bufferSize;

  // now convert the spline data to the new format
  for (int32_t iSector = 0; iSector < mGeo.getNumberOfSectors(); iSector++) {
    bool isAside = (iSector < mGeo.getNumberOfSectorsA());
    for (int32_t iRow = 0; iRow < mGeo.getNumberOfRows(); iRow++) {

      RowInfo& rowInfo = getRowInfo(iRow);
      const auto& spline = mScenarioPtr[rowInfo.splineScenarioID];

      int nSplineDimensions[3] = {3, 1, 2};

      for (int iSpline = 0; iSpline < 3; iSpline++) {
        int nDim = nSplineDimensions[iSpline];
        int nKnotParameters = 4 * nDim;
        auto* data = getCorrectionData(iSector, iRow, iSpline);

        // lambda to swap parameters at two knots
        auto swapKnots = [&](int i1, int j1, int i2, int j2) {
          auto k1 = spline.getKnotIndex(i1, j1);
          auto k2 = spline.getKnotIndex(i2, j2);
          for (int ipar = 0; ipar < nKnotParameters; ipar++) {
            std::swap(data[nKnotParameters * k1 + ipar], data[nKnotParameters * k2 + ipar]);
          }
        };

        // reorder knots for the A side U == old U, V == - old V
        if (isAside) {
          for (int32_t i = 0; i < spline.getGridX1().getNumberOfKnots(); i++) {
            for (int32_t j = 0; j < spline.getGridX2().getNumberOfKnots() / 2; j++) {
              swapKnots(i, j, i, spline.getGridX2().getNumberOfKnots() - 1 - j);
            }
          }
        } else { // reorder knots for the C side U == - old U, V == - old V
          for (int32_t i = 0; i < spline.getGridX1().getNumberOfKnots() / 2; i++) {
            for (int32_t j = 0; j < spline.getGridX2().getNumberOfKnots(); j++) {
              swapKnots(i, j, spline.getGridX1().getNumberOfKnots() - 1 - i, spline.getGridX2().getNumberOfKnots() - 1 - j);
            }
          }
        }

        // correct sign of the parameters due to the coordinate swaps

        for (int32_t iKnot = 0; iKnot < spline.getNumberOfKnots(); iKnot++) {
          // new grid directions for all corrections
          for (int iDim = 0; iDim < nDim; iDim++) {
            if (isAside) {
              data[nKnotParameters * iKnot + nDim * 1 + iDim] *= -1; // invert Z derivatives on A side
              data[nKnotParameters * iKnot + nDim * 3 + iDim] *= -1; // invert cross derivatives on A side
            } else {
              data[nKnotParameters * iKnot + nDim * 1 + iDim] *= -1; // invert Z derivatives on C side
              data[nKnotParameters * iKnot + nDim * 2 + iDim] *= -1; // invert Y derivatives on C side
            }
          }
          // new correction directions
          if (iSpline == 0) { // dX,dU,dV -> dX,dY,dZ
            if (isAside) {
              data[nKnotParameters * iKnot + nDim * 0 + 2] *= -1; // invert correction in Z
              data[nKnotParameters * iKnot + nDim * 1 + 2] *= -1; // invert correction in Z Z-derivative
              data[nKnotParameters * iKnot + nDim * 2 + 2] *= -1; // invert correction in Z Y-derivative
              data[nKnotParameters * iKnot + nDim * 3 + 2] *= -1; // invert correction in Z cross derivative
            } else {
              data[nKnotParameters * iKnot + nDim * 0 + 1] *= -1; // invert correction in Y
              data[nKnotParameters * iKnot + nDim * 1 + 1] *= -1; // invert correction in Y Z-derivative
              data[nKnotParameters * iKnot + nDim * 2 + 1] *= -1; // invert correction in Y Y-derivative
              data[nKnotParameters * iKnot + nDim * 3 + 1] *= -1; // invert correction in Y cross derivative
            }
          } else if (iSpline == 2) { // dU,dV at real U,V -> dY,dZ at real Y,Z
            if (isAside) {
              data[nKnotParameters * iKnot + nDim * 0 + 1] *= -1; // invert correction in Z
              data[nKnotParameters * iKnot + nDim * 1 + 1] *= -1; // invert correction in Z Z-derivative
              data[nKnotParameters * iKnot + nDim * 2 + 1] *= -1; // invert correction in Z Y-derivative
              data[nKnotParameters * iKnot + nDim * 3 + 1] *= -1; // invert correction in Z cross derivative
            } else {
              data[nKnotParameters * iKnot + nDim * 0 + 0] *= -1; // invert correction in Y
              data[nKnotParameters * iKnot + nDim * 1 + 0] *= -1; // invert correction in Y Z-derivative
              data[nKnotParameters * iKnot + nDim * 2 + 0] *= -1; // invert correction in Y Y-derivative
              data[nKnotParameters * iKnot + nDim * 3 + 0] *= -1; // invert correction in Y cross derivative
            }
          }
        }

      } // iSpline
    } // iRow
  } // iSector

  // set the class version to the current one
  mClassVersion = 4;
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
  mCorrectionData[0] = relocatePointer(oldBuffer, newBuffer, mCorrectionData[0]);
  mCorrectionData[1] = relocatePointer(oldBuffer, newBuffer, mCorrectionData[1]);
  mCorrectionData[2] = relocatePointer(oldBuffer, newBuffer, mCorrectionData[2]);

  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

void TPCFastSpaceChargeCorrection::print() const
{
  LOG(info) << " TPC Correction: ";
  mGeo.print();
  LOG(info) << "  mNumberOfScenarios = " << mNumberOfScenarios;
  LOG(info) << "  mTimeStamp = " << mTimeStamp;
  LOG(info) << "  mSectorDataSizeBytes = " << mSectorDataSizeBytes[0] << " " << mSectorDataSizeBytes[1] << " " << mSectorDataSizeBytes[2];

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
        const SplineType& spline = getSplineForRow(ir);
        const float* d = getCorrectionData(is, ir);
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

void TPCFastSpaceChargeCorrection::startConstruction(const TPCFastTransformGeo& geo, int32_t numberOfSplineScenarios)
{
  /// Starts the construction procedure, reserves temporary memory

  FlatObject::startConstruction();

  assert((geo.isConstructed()) && (numberOfSplineScenarios > 0));

  mGeo = geo;
  mNumberOfScenarios = numberOfSplineScenarios;

  releaseConstructionMemory();

  mConstructionScenarios = new SplineType[mNumberOfScenarios];

  assert(mConstructionScenarios != nullptr);

  for (int32_t i = 0; i < mGeo.getNumberOfRows(); i++) {
    auto& row = mRowInfos[i];
    row.splineScenarioID = -1;
    row.gridReal = {};
    row.gridMeasured = {};
    row.dataOffsetBytes[0] = 0;
    row.dataOffsetBytes[1] = 0;
    row.dataOffsetBytes[2] = 0;
  }

  for (int32_t i = 0; i < mNumberOfScenarios; i++) {
    mConstructionScenarios[i].destroy();
  }

  mTimeStamp = -1;

  mScenarioPtr = nullptr;
  for (int32_t s = 0; s < 3; s++) {
    mCorrectionData[s] = nullptr;
    mSectorDataSizeBytes[s] = 0;
  }
  mClassVersion = 4;
}

void TPCFastSpaceChargeCorrection::setRowScenarioID(int32_t iRow, int32_t iScenario)
{
  /// Initializes a TPC row
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(iRow >= 0 && iRow < mGeo.getNumberOfRows() && iScenario >= 0 && iScenario < mNumberOfScenarios);
  auto& row = getRowInfo(iRow);
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

  for (int32_t j = 0; j < mGeo.getNumberOfRows(); j++) {
    [[maybe_unused]] RowInfo& row = getRowInfo(j);
    assert(row.splineScenarioID >= 0);
    assert(row.splineScenarioID < mNumberOfScenarios);
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
  size_t correctionDataOffset[3];
  for (int32_t is = 0; is < 3; is++) {
    bufferSize = alignSize(bufferSize, SplineType::getParameterAlignmentBytes());
    correctionDataOffset[is] = bufferSize;
    mSectorDataSizeBytes[is] = 0;
    for (int32_t j = 0; j < mGeo.getNumberOfRows(); j++) {
      RowInfo& row = getRowInfo(j);
      row.dataOffsetBytes[is] = mSectorDataSizeBytes[is];
      const SplineType& spline = mConstructionScenarios[row.splineScenarioID];
      if (is == 0) {
        const SplineTypeXYZ& splineXYZ = reinterpret_cast<const SplineTypeXYZ&>(spline);
        mSectorDataSizeBytes[is] += splineXYZ.getSizeOfParameters();
      } else if (is == 1) {
        const SplineTypeInvX& splineInvX = reinterpret_cast<const SplineTypeInvX&>(spline);
        mSectorDataSizeBytes[is] += splineInvX.getSizeOfParameters();
      } else if (is == 2) {
        const SplineTypeInvYZ& splineInvYZ = reinterpret_cast<const SplineTypeInvYZ&>(spline);
        mSectorDataSizeBytes[is] += splineInvYZ.getSizeOfParameters();
      }
      mSectorDataSizeBytes[is] = alignSize(mSectorDataSizeBytes[is], SplineType::getParameterAlignmentBytes());
    }
    bufferSize += mSectorDataSizeBytes[is] * mGeo.getNumberOfSectors();
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
    mCorrectionData[is] = reinterpret_cast<char*>(mFlatBufferPtr + correctionDataOffset[is]);
  }
  releaseConstructionMemory();

  mTimeStamp = -1;

  setNoCorrection();
}

GPUd() void TPCFastSpaceChargeCorrection::setNoCorrection()
{
  // initialise all corrections to 0.

  for (int32_t row = 0; row < mGeo.getNumberOfRows(); row++) {
    const SplineType& spline = getSplineForRow(row);
    RowInfo& info = getRowInfo(row);
    float y0 = mGeo.getRowInfo(row).getYmin();
    float yScale = spline.getGridX1().getUmax() / mGeo.getRowInfo(row).getYwidth();
    float z0 = 0.;
    float zScale = spline.getGridX2().getUmax() / mGeo.getTPCzLength();
    info.gridMeasured.set(y0, yScale, z0, zScale, mGeo.getTPCzLength(), mGeo.getTPCzLength());
    info.gridReal = info.gridMeasured;
  } // row

  for (int32_t sector = 0; sector < mGeo.getNumberOfSectors(); sector++) {
    for (int32_t row = 0; row < mGeo.getNumberOfRows(); row++) {
      for (int32_t is = 0; is < 3; is++) {
        float* data = getCorrectionData(sector, row, is);
        int32_t nPar = 0;
        if (is == 0) {
          nPar = getSplineForRow(row).getNumberOfParameters();
        } else if (is == 1) {
          nPar = getSplineInvXforRow(row).getNumberOfParameters();
        } else if (is == 2) {
          nPar = getSplineInvYZforRow(row).getNumberOfParameters();
        }
        for (int32_t i = 0; i < nPar; i++) {
          data[i] = 0.f;
        }
      }
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

    MaxValue maxDsector[3];
    for (int32_t row = 0; row < mGeo.getNumberOfRows(); row++) {
      double x = mGeo.getRowInfo(row).x;
      auto [y0, y1] = mGeo.getRowInfo(row).getYrange();
      auto [z0, z1] = mGeo.getZrange(sector);

      double stepY = (y1 - y0) / 100.;
      double stepZ = (z1 - z0) / 100.;
      MaxValue maxDrow[3];
      for (double y = y0; y < y1; y += stepY) {
        for (double z = z0; z < z1; z += stepZ) {
          float dx, dy, dz;
          getCorrectionLocal(sector, row, y, z, dx, dy, dz);
          double realX = x + dx;
          double realY = y + dy;
          double realZ = z + dz;
          if (!isLocalInsideGrid(sector, row, y, z) || !isRealLocalInsideGrid(sector, row, realY, realZ)) {
            continue;
          }
          double r2 = realX * realX + realY * realY;
          if (realY < y0 || realY > y1 ||
              realZ < z0 || realZ > z1 ||
              r2 < tpcR2min || r2 > tpcR2max) {
            continue;
          }
          float dxr = getCorrectionXatRealYZ(sector, row, realY, realZ);
          float dyr, dzr;
          getCorrectionYZatRealYZ(sector, row, realY, realZ, dyr, dzr);
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
