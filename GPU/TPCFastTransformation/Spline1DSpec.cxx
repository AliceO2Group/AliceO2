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

/// \file  Spline1DSpec.cxx
/// \brief Implementation of Spline1DContainer & Spline1DSpec classes
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "Spline1DSpec.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "Spline1DHelper.h"
#include "TFile.h"
#include "GPUCommonMath.h"
templateClassImp(GPUCA_NAMESPACE::gpu::Spline1DContainer);
templateClassImp(GPUCA_NAMESPACE::gpu::Spline1DSpec);
#endif

using namespace std;
using namespace GPUCA_NAMESPACE::gpu;

#if !defined(GPUCA_GPUCODE)

template <class DataT>
void Spline1DContainer<DataT>::recreate(int nYdim, int numberOfKnots)
{
  /// Constructor for a regular spline
  /// \param numberOfKnots     Number of knots

  if (numberOfKnots < 2) {
    numberOfKnots = 2;
  }

  std::vector<int> knots(numberOfKnots);
  for (int i = 0; i < numberOfKnots; i++) {
    knots[i] = i;
  }
  recreate(nYdim, numberOfKnots, knots.data());
}

template <class DataT>
void Spline1DContainer<DataT>::recreate(int nYdim, int numberOfKnots, const int inputKnots[])
{
  /// Main constructor for an irregular spline
  ///
  /// Number of created knots may differ from the input values:
  /// - Duplicated knots will be deleted
  /// - At least 2 knots will be created
  ///
  /// \param numberOfKnots     Number of knots in knots[] array
  /// \param knots             Array of relative knot positions (integer values)
  ///

  FlatObject::startConstruction();

  mYdim = (nYdim >= 0) ? nYdim : 0;

  std::vector<int> knotU;

  { // sort knots
    std::vector<int> tmp;
    for (int i = 0; i < numberOfKnots; i++) {
      tmp.push_back(inputKnots[i]);
    }
    std::sort(tmp.begin(), tmp.end());

    knotU.push_back(0); //  first knot at 0

    for (unsigned int i = 1; i < tmp.size(); ++i) {
      int u = tmp[i] - tmp[0];
      if (knotU.back() < u) { // remove duplicated knots
        knotU.push_back(u);
      }
    }
    if (knotU.back() < 1) { // there is only one knot at u=0, add the second one at u=1
      knotU.push_back(1);
    }
  }

  mNumberOfKnots = knotU.size();
  mUmax = knotU.back();
  mXmin = 0.;
  mXtoUscale = 1.;

  const int uToKnotMapOffset = mNumberOfKnots * sizeof(Knot);
  int parametersOffset = uToKnotMapOffset + (mUmax + 1) * sizeof(int);
  int bufferSize = parametersOffset;
  if (mYdim > 0) {
    parametersOffset = alignSize(bufferSize, getParameterAlignmentBytes());
    bufferSize = parametersOffset + getSizeOfParameters();
  }

  FlatObject::finishConstruction(bufferSize);

  mUtoKnotMap = reinterpret_cast<int*>(mFlatBufferPtr + uToKnotMapOffset);
  mParameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);

  for (int i = 0; i < getNumberOfParameters(); i++) {
    mParameters[i] = 0;
  }

  Knot* s = getKnots();

  for (int i = 0; i < mNumberOfKnots; i++) {
    s[i].u = knotU[i];
  }

  for (int i = 0; i < mNumberOfKnots - 1; i++) {
    s[i].Li = 1. / (s[i + 1].u - s[i].u); // do division in double
  }

  s[mNumberOfKnots - 1].Li = 0.; // the value will not be used, we define it for consistency

  // Set up the map (integer U) -> (knot index)

  int* map = getUtoKnotMap();

  const int iKnotMax = mNumberOfKnots - 2;

  //
  // With iKnotMax=nKnots-2 we map the U==Umax coordinate to the last [nKnots-2, nKnots-1] segment.
  // This trick allows one to avoid a special condition for this edge case.
  // Any U from [0,Umax] is mapped to some knot_i such, that the next knot_i+1 always exist
  //

  for (int u = 0, iKnot = 0; u <= mUmax; u++) {
    if ((knotU[iKnot + 1] == u) && (iKnot < iKnotMax)) {
      iKnot = iKnot + 1;
    }
    map[u] = iKnot;
  }
}

#endif //GPUCA_GPUCODE

template <class DataT>
void Spline1DContainer<DataT>::print() const
{
  printf(" Spline 1D: \n");
  printf("  mNumberOfKnots = %d \n", mNumberOfKnots);
  printf("  mUmax = %d\n", mUmax);
  printf("  mUtoKnotMap = %p \n", (void*)mUtoKnotMap);
  printf("  knots: ");
  for (int i = 0; i < mNumberOfKnots; i++) {
    printf("%d ", (int)getKnot(i).u);
  }
  printf("\n");
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

template <class DataT>
void Spline1DContainer<DataT>::approximateFunction(
  double xMin, double xMax,
  std::function<void(double x, double f[])> F,
  int nAxiliaryDataPoints)
{
  /// approximate a function F with this spline
  Spline1DHelper<DataT> helper;
  helper.approximateFunction(*reinterpret_cast<Spline1D<DataT>*>(this), xMin, xMax, F, nAxiliaryDataPoints);
}

#ifndef GPUCA_ALIROOT_LIB
template <class DataT>
int Spline1DContainer<DataT>::writeToFile(TFile& outf, const char* name)
{
  /// write a class object to the file
  return FlatObject::writeToFile(*this, outf, name);
}

template <class DataT>
Spline1DContainer<DataT>* Spline1DContainer<DataT>::readFromFile(
  TFile& inpf, const char* name)
{
  /// read a class object from the file
  return FlatObject::readFromFile<Spline1DContainer<DataT>>(inpf, name);
}

#endif
#endif

#if !defined(GPUCA_GPUCODE)

template <class DataT>
void Spline1DContainer<DataT>::cloneFromObject(const Spline1DContainer<DataT>& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mYdim = obj.mYdim;
  mNumberOfKnots = obj.mNumberOfKnots;
  mUmax = obj.mUmax;
  mXmin = obj.mXmin;
  mXtoUscale = obj.mXtoUscale;
  mUtoKnotMap = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mUtoKnotMap);
  mParameters = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mParameters);
}

template <class DataT>
void Spline1DContainer<DataT>::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = mFlatBufferPtr;
  mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}
#endif // GPUCA_GPUCODE

template <class DataT>
void Spline1DContainer<DataT>::destroy()
{
  /// See FlatObject for description
  mNumberOfKnots = 0;
  mUmax = 0;
  mYdim = 0;
  mXmin = 0.;
  mXtoUscale = 1.;
  mUtoKnotMap = nullptr;
  mParameters = nullptr;
  FlatObject::destroy();
}

template <class DataT>
void Spline1DContainer<DataT>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description

  FlatObject::setActualBufferAddress(actualFlatBufferPtr);

  const int uToKnotMapOffset = mNumberOfKnots * sizeof(Knot);
  mUtoKnotMap = reinterpret_cast<int*>(mFlatBufferPtr + uToKnotMapOffset);
  int parametersOffset = uToKnotMapOffset + (mUmax + 1) * sizeof(int);
  if (mYdim > 0) {
    parametersOffset = alignSize(parametersOffset, getParameterAlignmentBytes());
  }
  mParameters = reinterpret_cast<DataT*>(mFlatBufferPtr + parametersOffset);
}

template <class DataT>
void Spline1DContainer<DataT>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  mUtoKnotMap = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mUtoKnotMap);
  mParameters = relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mParameters);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)
template <class DataT>
int Spline1DContainer<DataT>::test(const bool draw, const bool drawDataPoints)
{
  return Spline1DHelper<DataT>::test(draw, drawDataPoints);
}
#endif // GPUCA_GPUCODE

template class GPUCA_NAMESPACE::gpu::Spline1DContainer<float>;
template class GPUCA_NAMESPACE::gpu::Spline1DContainer<double>;
template class GPUCA_NAMESPACE::gpu::Spline1DSpec<float, 0, 2>;
template class GPUCA_NAMESPACE::gpu::Spline1DSpec<double, 0, 2>;
