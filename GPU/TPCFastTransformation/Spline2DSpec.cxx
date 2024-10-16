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

/// \file  Spline2DSpec.cxx
/// \brief Implementation of Spline2DSpec class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "Rtypes.h"
#endif

#include "Spline2DSpec.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "TRandom.h"
#include "Riostream.h"
#include "TMath.h"
#include "Spline2DHelper.h"
#include "TCanvas.h"
#include "TNtuple.h"
#include "TFile.h"
#include "GPUCommonMath.h"

templateClassImp(GPUCA_NAMESPACE::gpu::Spline2DContainer);
templateClassImp(GPUCA_NAMESPACE::gpu::Spline2DSpec);

#endif

using namespace std;
using namespace GPUCA_NAMESPACE::gpu;

template <typename DataT>
void Spline2DContainer<DataT>::destroy()
{
  /// See FlatObject for description
  mGridX1.destroy();
  mGridX2.destroy();
  mYdim = 0;
  mParameters = nullptr;
  FlatObject::destroy();
}

template <typename DataT>
void Spline2DContainer<DataT>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description

  FlatObject::setActualBufferAddress(actualFlatBufferPtr);

  const size_t u2Offset = alignSize(mGridX1.getFlatBufferSize(), mGridX2.getBufferAlignmentBytes());
  int32_t parametersOffset = u2Offset;
  // int32_t bufferSize = parametersOffset;
  mParameters = nullptr;

  parametersOffset = alignSize(u2Offset + mGridX2.getFlatBufferSize(), getParameterAlignmentBytes());
  //bufferSize = parametersOffset + getSizeOfParameters();
  mParameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);

  mGridX1.setActualBufferAddress(mFlatBufferPtr);
  mGridX2.setActualBufferAddress(mFlatBufferPtr + u2Offset);
}

template <typename DataT>
void Spline2DContainer<DataT>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  char* bufferU = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mGridX1.getFlatBufferPtr());
  char* bufferV = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mGridX2.getFlatBufferPtr());
  mGridX1.setFutureBufferAddress(bufferU);
  mGridX2.setFutureBufferAddress(bufferV);
  mParameters = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mParameters);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

template <typename DataT>
void Spline2DContainer<DataT>::print() const
{
  printf(" Irregular Spline 2D: \n");
  printf(" grid U1: \n");
  mGridX1.print();
  printf(" grid U2: \n");
  mGridX2.print();
}

#if !defined(GPUCA_GPUCODE)

template <typename DataT>
void Spline2DContainer<DataT>::cloneFromObject(const Spline2DContainer<DataT>& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;

  FlatObject::cloneFromObject(obj, newFlatBufferPtr);

  mYdim = obj.mYdim;
  char* bufferU = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGridX1.getFlatBufferPtr());
  char* bufferV = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mGridX2.getFlatBufferPtr());

  mGridX1.cloneFromObject(obj.mGridX1, bufferU);
  mGridX2.cloneFromObject(obj.mGridX2, bufferV);
  mParameters = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mParameters);
}

template <typename DataT>
void Spline2DContainer<DataT>::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = mFlatBufferPtr;
  mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}

template <typename DataT>
void Spline2DContainer<DataT>::recreate(
  int32_t nYdim,
  int32_t numberOfKnotsU1, const int32_t knotsU1[], int32_t numberOfKnotsU2, const int32_t knotsU2[])
{
  /// Constructor for an irregular spline

  mYdim = nYdim;
  FlatObject::startConstruction();

  mGridX1.recreate(0, numberOfKnotsU1, knotsU1);
  mGridX2.recreate(0, numberOfKnotsU2, knotsU2);

  const size_t u2Offset = alignSize(mGridX1.getFlatBufferSize(), mGridX2.getBufferAlignmentBytes());
  int32_t parametersOffset = u2Offset + mGridX2.getFlatBufferSize();
  int32_t bufferSize = parametersOffset;
  mParameters = nullptr;

  parametersOffset = alignSize(bufferSize, getParameterAlignmentBytes());
  bufferSize = parametersOffset + getSizeOfParameters();

  FlatObject::finishConstruction(bufferSize);

  mGridX1.moveBufferTo(mFlatBufferPtr);
  mGridX2.moveBufferTo(mFlatBufferPtr + u2Offset);

  mParameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);
  for (int32_t i = 0; i < getNumberOfParameters(); i++) {
    mParameters[i] = 0;
  }
}

template <typename DataT>
void Spline2DContainer<DataT>::recreate(int32_t nYdim,
                                        int32_t numberOfKnotsU1, int32_t numberOfKnotsU2)
{
  /// Constructor for a regular spline

  mYdim = nYdim;
  FlatObject::startConstruction();

  mGridX1.recreate(0, numberOfKnotsU1);
  mGridX2.recreate(0, numberOfKnotsU2);

  const size_t u2Offset = alignSize(mGridX1.getFlatBufferSize(), mGridX2.getBufferAlignmentBytes());
  int32_t parametersOffset = u2Offset + mGridX2.getFlatBufferSize();
  int32_t bufferSize = parametersOffset;
  mParameters = nullptr;

  parametersOffset = alignSize(bufferSize, getParameterAlignmentBytes());
  bufferSize = parametersOffset + getSizeOfParameters();

  FlatObject::finishConstruction(bufferSize);

  mGridX1.moveBufferTo(mFlatBufferPtr);
  mGridX2.moveBufferTo(mFlatBufferPtr + u2Offset);

  mParameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);
  for (int32_t i = 0; i < getNumberOfParameters(); i++) {
    mParameters[i] = 0;
  }
}

#endif // GPUCA_GPUCODE

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation

template <typename DataT>
void Spline2DContainer<DataT>::approximateFunction(
  double x1Min, double x1Max, double x2Min, double x2Max,
  std::function<void(double x1, double x2, double f[])> F,
  int32_t nAuxiliaryDataPointsX1, int32_t nAuxiliaryDataPointsX2)
{
  /// approximate a function F with this spline
  Spline2DHelper<DataT> helper;
  helper.approximateFunction(*reinterpret_cast<Spline2D<DataT>*>(this), x1Min, x1Max, x2Min, x2Max, F, nAuxiliaryDataPointsX1, nAuxiliaryDataPointsX2);
}

template <typename DataT>
void Spline2DContainer<DataT>::approximateFunctionViaDataPoints(
  double x1Min, double x1Max, double x2Min, double x2Max,
  std::function<void(double x1, double x2, double f[])> F,
  int32_t nAuxiliaryDataPointsX1, int32_t nAuxiliaryDataPointsX2)
{
  /// approximate a function F with this spline
  Spline2DHelper<DataT> helper;
  helper.approximateFunctionViaDataPoints(*reinterpret_cast<Spline2D<DataT>*>(this), x1Min, x1Max, x2Min, x2Max, F, nAuxiliaryDataPointsX1, nAuxiliaryDataPointsX2);
}

#ifndef GPUCA_ALIROOT_LIB
template <typename DataT>
int32_t Spline2DContainer<DataT>::writeToFile(TFile& outf, const char* name)
{
  /// write a class object to the file
  return FlatObject::writeToFile(*this, outf, name);
}

template <typename DataT>
Spline2DContainer<DataT>* Spline2DContainer<DataT>::readFromFile(
  TFile& inpf, const char* name)
{
  /// read a class object from the file
  return FlatObject::readFromFile<Spline2DContainer<DataT>>(inpf, name);
}

template <typename DataT>
int32_t Spline2DContainer<DataT>::test(const bool draw, const bool drawDataPoints)
{
  return Spline2DHelper<DataT>::test(draw, drawDataPoints);
}
#endif

#endif // GPUCA_GPUCODE && !GPUCA_STANDALONE

template class GPUCA_NAMESPACE::gpu::Spline2DContainer<float>;
template class GPUCA_NAMESPACE::gpu::Spline2DContainer<double>;
