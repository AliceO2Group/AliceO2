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

#if !defined(GPUCA_STANDALONE) // code invisible in the standalone compilation
#include "Rtypes.h"
#endif

#include "Spline2DSpec.h"

#include <iostream>

#if !defined(GPUCA_STANDALONE) // code invisible in the standalone compilation
#include "TRandom.h"
#include "Riostream.h"
#include "TMath.h"
#include "Spline2DHelper.h"
#include "TFile.h"
#include "GPUCommonMath.h"

templateClassImp(o2::gpu::Spline2DContainerBase);
templateClassImp(o2::gpu::Spline2DSpec);

#endif

using namespace std;
using namespace o2::gpu;

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::destroy()
{
  /// See FlatObject for description
  mGridX1.destroy();
  mGridX2.destroy();
  mYdim = 0;
  mParameters = nullptr;
  FlatBase::destroy();
}

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description

  FlatBase::setActualBufferAddress(actualFlatBufferPtr);

  const size_t u2Offset = this->alignSize(mGridX1.getFlatBufferSize(), mGridX2.getBufferAlignmentBytes());
  int32_t parametersOffset = u2Offset;
  mParameters = nullptr;

  parametersOffset = this->alignSize(u2Offset + mGridX2.getFlatBufferSize(), getParameterAlignmentBytes());
  mParameters = reinterpret_cast<DataT*>(this->mFlatBufferPtr + parametersOffset);

  mGridX1.setActualBufferAddress(this->mFlatBufferPtr);
  mGridX2.setActualBufferAddress(this->mFlatBufferPtr + u2Offset);
}

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  char* bufferU = FlatBase::relocatePointer(this->mFlatBufferPtr, futureFlatBufferPtr, mGridX1.getFlatBufferPtr());
  char* bufferV = FlatBase::relocatePointer(this->mFlatBufferPtr, futureFlatBufferPtr, mGridX2.getFlatBufferPtr());
  mGridX1.setFutureBufferAddress(bufferU);
  mGridX2.setFutureBufferAddress(bufferV);
  mParameters = FlatBase::relocatePointer(this->mFlatBufferPtr, futureFlatBufferPtr, mParameters);
  FlatBase::setFutureBufferAddress(futureFlatBufferPtr);
}

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::print() const
{
  printf(" Irregular Spline 2D: \n");
  printf(" grid U1: \n");
  mGridX1.print();
  printf(" grid U2: \n");
  mGridX2.print();
}

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::cloneFromObject(const Spline2DContainerBase<DataT, FlatBase>& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;

  FlatBase::cloneFromObject(obj, newFlatBufferPtr);

  mYdim = obj.mYdim;
  char* bufferU = FlatBase::relocatePointer(oldFlatBufferPtr, this->mFlatBufferPtr, obj.mGridX1.getFlatBufferPtr());
  char* bufferV = FlatBase::relocatePointer(oldFlatBufferPtr, this->mFlatBufferPtr, obj.mGridX2.getFlatBufferPtr());

  mGridX1.cloneFromObject(obj.mGridX1, bufferU);
  mGridX2.cloneFromObject(obj.mGridX2, bufferV);
  mParameters = FlatBase::relocatePointer(oldFlatBufferPtr, this->mFlatBufferPtr, obj.mParameters);
}

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  char* oldFlatBufferPtr = this->mFlatBufferPtr;
  FlatBase::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = this->mFlatBufferPtr;
  this->mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}

template <typename DataT, class FlatBase>
template <class OtherFlatBase>
void Spline2DContainerBase<DataT, FlatBase>::importFrom(const Spline2DContainerBase<DataT, OtherFlatBase>& src)
{
  /// Copy schema fields from a spline with a different FlatBase (e.g. FlatObject -> NoFlatObject).
  /// Grid pointers (mKnots, mUtoKnotMap) and mParameters are left null; call setActualBufferAddress() afterward.
  mYdim = src.getYdimensions();
  this->mFlatBufferSize = src.getFlatBufferSize();
  mGridX1.importFrom(src.getGridX1());
  mGridX2.importFrom(src.getGridX2());
  mParameters = nullptr;
}

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::recreate(
  int32_t nYdim,
  int32_t numberOfKnotsU1, const int32_t knotsU1[], int32_t numberOfKnotsU2, const int32_t knotsU2[])
{
  /// Constructor for an irregular spline

  mYdim = nYdim;
  FlatBase::startConstruction();

  mGridX1.recreate(0, numberOfKnotsU1, knotsU1);
  mGridX2.recreate(0, numberOfKnotsU2, knotsU2);

  const size_t u2Offset = this->alignSize(mGridX1.getFlatBufferSize(), mGridX2.getBufferAlignmentBytes());
  int32_t parametersOffset = u2Offset + mGridX2.getFlatBufferSize();
  int32_t bufferSize = parametersOffset;
  mParameters = nullptr;

  parametersOffset = this->alignSize(bufferSize, getParameterAlignmentBytes());
  bufferSize = parametersOffset + getSizeOfParameters();

  FlatBase::finishConstruction(bufferSize);

  mGridX1.moveBufferTo(this->mFlatBufferPtr);
  mGridX2.moveBufferTo(this->mFlatBufferPtr + u2Offset);

  mParameters = reinterpret_cast<DataT*>(this->mFlatBufferPtr + parametersOffset);
  for (int32_t i = 0; i < getNumberOfParameters(); i++) {
    mParameters[i] = 0;
  }
}

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::recreate(int32_t nYdim,
                                                  int32_t numberOfKnotsU1, int32_t numberOfKnotsU2)
{
  /// Constructor for a regular spline

  mYdim = nYdim;
  FlatBase::startConstruction();

  mGridX1.recreate(0, numberOfKnotsU1);
  mGridX2.recreate(0, numberOfKnotsU2);

  const size_t u2Offset = this->alignSize(mGridX1.getFlatBufferSize(), mGridX2.getBufferAlignmentBytes());
  int32_t parametersOffset = u2Offset + mGridX2.getFlatBufferSize();
  int32_t bufferSize = parametersOffset;
  mParameters = nullptr;

  parametersOffset = this->alignSize(bufferSize, getParameterAlignmentBytes());
  bufferSize = parametersOffset + getSizeOfParameters();

  FlatBase::finishConstruction(bufferSize);

  mGridX1.moveBufferTo(this->mFlatBufferPtr);
  mGridX2.moveBufferTo(this->mFlatBufferPtr + u2Offset);

  mParameters = reinterpret_cast<DataT*>(this->mFlatBufferPtr + parametersOffset);
  for (int32_t i = 0; i < getNumberOfParameters(); i++) {
    mParameters[i] = 0;
  }
}

#if !defined(GPUCA_STANDALONE) // code invisible in the standalone compilation

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::approximateFunction(
  double x1Min, double x1Max, double x2Min, double x2Max,
  std::function<void(double x1, double x2, double f[])> F,
  int32_t nAuxiliaryDataPointsX1, int32_t nAuxiliaryDataPointsX2)
{
  /// approximate a function F with this spline
  Spline2DHelper<DataT> helper;
  helper.approximateFunction(*reinterpret_cast<Spline2D<DataT>*>(this), x1Min, x1Max, x2Min, x2Max, F, nAuxiliaryDataPointsX1, nAuxiliaryDataPointsX2);
}

template <typename DataT, class FlatBase>
void Spline2DContainerBase<DataT, FlatBase>::approximateFunctionViaDataPoints(
  double x1Min, double x1Max, double x2Min, double x2Max,
  std::function<void(double x1, double x2, double f[])> F,
  int32_t nAuxiliaryDataPointsX1, int32_t nAuxiliaryDataPointsX2)
{
  /// approximate a function F with this spline
  Spline2DHelper<DataT> helper;
  helper.approximateFunctionViaDataPoints(*reinterpret_cast<Spline2D<DataT>*>(this), x1Min, x1Max, x2Min, x2Max, F, nAuxiliaryDataPointsX1, nAuxiliaryDataPointsX2);
}

template <typename DataT, class FlatBase>
int32_t Spline2DContainerBase<DataT, FlatBase>::writeToFile(TFile& outf, const char* name)
{
  /// write a class object to the file
  if constexpr (std::is_same_v<FlatBase, FlatObject>) {
    return FlatObject::writeToFile(*this, outf, name);
  } else {
    return -1;
  }
}

template <typename DataT, class FlatBase>
Spline2DContainerBase<DataT, FlatBase>* Spline2DContainerBase<DataT, FlatBase>::readFromFile(TFile& inpf, const char* name)
{
  /// read a class object from the file
  if constexpr (std::is_same_v<FlatBase, FlatObject>) {
    return FlatObject::readFromFile<Spline2DContainerBase<DataT, FlatBase>>(inpf, name);
  } else {
    return nullptr;
  }
}

template <typename DataT, class FlatBase>
int32_t Spline2DContainerBase<DataT, FlatBase>::test(const bool draw, const bool drawDataPoints)
{
  return Spline2DHelper<DataT>::test(draw, drawDataPoints);
}

#endif // !GPUCA_STANDALONE

template class o2::gpu::Spline2DContainerBase<float>;
template class o2::gpu::Spline2DContainerBase<double>;

// Explicit instantiations for NoFlatObject (used by TPCFastTransformPOD)
template class o2::gpu::Spline2DContainerBase<float, o2::gpu::NoFlatObject>;
template class o2::gpu::Spline2DContainerBase<double, o2::gpu::NoFlatObject>;
// importFrom instantiation for the FlatObject -> NoFlatObject conversion used in create()
template void o2::gpu::Spline2DContainerBase<float, o2::gpu::NoFlatObject>::importFrom<o2::gpu::FlatObject>(const o2::gpu::Spline2DContainerBase<float, o2::gpu::FlatObject>&);
template void o2::gpu::Spline2DContainerBase<double, o2::gpu::NoFlatObject>::importFrom<o2::gpu::FlatObject>(const o2::gpu::Spline2DContainerBase<double, o2::gpu::FlatObject>&);
