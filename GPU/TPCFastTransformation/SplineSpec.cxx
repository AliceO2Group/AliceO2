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

/// \file  SplineSpec.cxx
/// \brief Implementation of SplineSpec class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "Rtypes.h"
#endif

#include "SplineSpec.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "TRandom.h"
#include "Riostream.h"
#include "TMath.h"
#include "SplineHelper.h"
#include "TCanvas.h"
#include "TNtuple.h"
#include "TFile.h"
#include "GPUCommonMath.h"

templateClassImp(GPUCA_NAMESPACE::gpu::SplineContainer);
templateClassImp(GPUCA_NAMESPACE::gpu::SplineSpec);

#endif

using namespace std;
using namespace GPUCA_NAMESPACE::gpu;

template <typename DataT>
void SplineContainer<DataT>::destroy()
{
  /// See FlatObject for description
  mXdim = 0;
  mYdim = 0;
  mNknots = 0;
  mGrid = nullptr;
  mParameters = nullptr;
  FlatObject::destroy();
}

template <typename DataT>
void SplineContainer<DataT>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description

  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  mGrid = reinterpret_cast<Spline1D<DataT>*>(mFlatBufferPtr);
  int32_t offset = sizeof(*mGrid) * mXdim;
  for (int32_t i = 0; i < mXdim; i++) {
    offset = alignSize(offset, mGrid[i].getBufferAlignmentBytes());
    mGrid[i].setActualBufferAddress(mFlatBufferPtr + offset);
    offset += mGrid[i].getFlatBufferSize();
  }
  offset = alignSize(offset, getParameterAlignmentBytes());
  mParameters = reinterpret_cast<DataT*>(mFlatBufferPtr + offset);
}

template <typename DataT>
void SplineContainer<DataT>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  mParameters = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mParameters);
  for (int32_t i = 0; i < mXdim; i++) {
    char* buffer = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mGrid[i].getFlatBufferPtr());
    mGrid[i].setFutureBufferAddress(buffer);
  }
  mGrid = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mGrid);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

template <typename DataT>
void SplineContainer<DataT>::print() const
{
  printf(" Irregular Spline %dD->%dD: \n", mXdim, mYdim);
  for (int32_t i = 0; i < mXdim; i++) {
    printf(" grid X%d: \n", i);
    mGrid[i].print();
  }
}

#if !defined(GPUCA_GPUCODE)

template <typename DataT>
void SplineContainer<DataT>::cloneFromObject(const SplineContainer<DataT>& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mXdim = obj.mXdim;
  mYdim = obj.mYdim;
  mNknots = obj.mNknots;

  Spline1D<DataT>* newGrid = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGrid);
  for (int32_t i = 0; i < mXdim; i++) {
    char* buffer = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGrid[i].getFlatBufferPtr());
    newGrid[i].cloneFromObject(obj.mGrid[i], buffer);
  }
  mGrid = newGrid;
  mParameters = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mParameters);
}

template <typename DataT>
void SplineContainer<DataT>::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = mFlatBufferPtr;
  mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}

template <typename DataT>
void SplineContainer<DataT>::recreate(
  int32_t nXdim, int32_t nYdim, const int32_t numberOfKnots[/* nXdim */], const int32_t* const knots[/* nXdim */])
{
  /// Constructor for an irregular spline

  mXdim = nXdim;
  mYdim = nYdim;
  FlatObject::startConstruction();

  Spline1D<DataT> vGrids[mXdim];

  mNknots = 1;
  for (int32_t i = 0; i < mXdim; i++) {
    if (knots) {
      vGrids[i].recreate(0, numberOfKnots[i], knots[i]);
    } else if (numberOfKnots) {
      vGrids[i].recreate(0, numberOfKnots[i]);
    } else {
      vGrids[i].recreate(0, 2);
    }
    mNknots *= vGrids[i].getNumberOfKnots();
  }

  int32_t offset = sizeof(Spline1D<DataT>) * mXdim;

  for (int32_t i = 0; i < mXdim; i++) {
    offset = alignSize(offset, vGrids[i].getBufferAlignmentBytes());
    offset += vGrids[i].getFlatBufferSize();
  }

  offset = alignSize(offset, getParameterAlignmentBytes());
  offset += getSizeOfParameters();

  FlatObject::finishConstruction(offset);

  mGrid = reinterpret_cast<Spline1D<DataT>*>(mFlatBufferPtr);

  offset = sizeof(Spline1D<DataT>) * mXdim;

  for (int32_t i = 0; i < mXdim; i++) {
    new (&mGrid[i]) Spline1D<DataT>; // constructor
    offset = alignSize(offset, mGrid[i].getBufferAlignmentBytes());
    mGrid[i].cloneFromObject(vGrids[i], mFlatBufferPtr + offset);
    offset += mGrid[i].getFlatBufferSize();
  }

  offset = alignSize(offset, getParameterAlignmentBytes());
  mParameters = reinterpret_cast<DataT*>(mFlatBufferPtr + offset);
  offset += getSizeOfParameters();

  for (int32_t i = 0; i < getNumberOfParameters(); i++) {
    mParameters[i] = 0;
  }
}

template <typename DataT>
void SplineContainer<DataT>::recreate(
  int32_t nXdim, int32_t nYdim, const int32_t numberOfKnots[/* nXdim */])
{
  /// Constructor for a regular spline
  recreate(nXdim, nYdim, numberOfKnots, nullptr);
}

#endif // GPUCA_GPUCODE

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation

template <typename DataT>
void SplineContainer<DataT>::
  approximateFunction(const double xMin[/* mXdim */], const double xMax[/* mXdim */],
                      std::function<void(const double x[/* mXdim */], double f[/*mYdim*/])> F,
                      const int32_t nAuxiliaryDataPoints[/* mXdim */])
{
  /// approximate a function F with this spline
  SplineHelper<DataT> helper;
  helper.approximateFunction(*reinterpret_cast<Spline<DataT>*>(this), xMin, xMax, F, nAuxiliaryDataPoints);
}

#ifndef GPUCA_ALIROOT_LIB
template <typename DataT>
int32_t SplineContainer<DataT>::writeToFile(TFile& outf, const char* name)
{
  /// write a class object to the file
  return FlatObject::writeToFile(*this, outf, name);
}

template <typename DataT>
SplineContainer<DataT>* SplineContainer<DataT>::readFromFile(
  TFile& inpf, const char* name)
{
  /// read a class object from the file
  return FlatObject::readFromFile<SplineContainer<DataT>>(inpf, name);
}

template <typename DataT>
int32_t SplineContainer<DataT>::test(const bool draw, const bool drawDataPoints)
{
  return SplineHelper<DataT>::test(draw, drawDataPoints);
}
#endif

#endif // GPUCA_GPUCODE && !GPUCA_STANDALONE

template class GPUCA_NAMESPACE::gpu::SplineContainer<float>;
template class GPUCA_NAMESPACE::gpu::SplineContainer<double>;
