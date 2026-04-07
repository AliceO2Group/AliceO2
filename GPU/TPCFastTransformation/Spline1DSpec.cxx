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

#include <iostream>
#include <algorithm>

#if !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "Spline1DHelper.h"
#include "TFile.h"
#include "GPUCommonMath.h"
templateClassImp(o2::gpu::Spline1DContainer);
templateClassImp(o2::gpu::Spline1DSpec);
#endif

using namespace std;
using namespace o2::gpu;

template <class DataT, class FlatBase>
void Spline1DContainerBase<DataT, FlatBase>::recreate(int32_t nYdim, int32_t numberOfKnots)
{
  /// Constructor for a regular spline
  /// \param numberOfKnots     Number of knots

  if (numberOfKnots < 2) {
    numberOfKnots = 2;
  }

  std::vector<int32_t> knots(numberOfKnots);
  for (int32_t i = 0; i < numberOfKnots; i++) {
    knots[i] = i;
  }
  recreate(nYdim, numberOfKnots, knots.data());
}

template <class DataT, class FlatBase>
void Spline1DContainerBase<DataT, FlatBase>::recreate(int32_t nYdim, int32_t numberOfKnots, const int32_t inputKnots[])
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

  FlatBase::startConstruction();

  mYdim = (nYdim >= 0) ? nYdim : 0;

  std::vector<int32_t> knotU;

  { // sort knots
    std::vector<int32_t> tmp;
    for (int32_t i = 0; i < numberOfKnots; i++) {
      tmp.push_back(inputKnots[i]);
    }
    std::sort(tmp.begin(), tmp.end());

    knotU.push_back(0); //  first knot at 0

    for (uint32_t i = 1; i < tmp.size(); ++i) {
      int32_t u = tmp[i] - tmp[0];
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

  const int32_t uToKnotMapOffset = mNumberOfKnots * sizeof(Knot);
  int32_t parametersOffset = uToKnotMapOffset + (mUmax + 1) * sizeof(int32_t);
  int32_t bufferSize = parametersOffset;
  if (mYdim > 0) {
    parametersOffset = this->alignSize(bufferSize, getParameterAlignmentBytes());
    bufferSize = parametersOffset + getSizeOfParameters();
  }

  FlatBase::finishConstruction(bufferSize);

  mUtoKnotMap = reinterpret_cast<int32_t*>(this->mFlatBufferPtr + uToKnotMapOffset);
  mParameters = reinterpret_cast<DataT*>(this->mFlatBufferPtr + parametersOffset);

  for (int32_t i = 0; i < getNumberOfParameters(); i++) {
    mParameters[i] = 0;
  }

  Knot* s = getKnots();

  for (int32_t i = 0; i < mNumberOfKnots; i++) {
    s[i].u = knotU[i];
  }

  for (int32_t i = 0; i < mNumberOfKnots - 1; i++) {
    s[i].Li = 1. / (s[i + 1].u - s[i].u); // do division in double
  }

  s[mNumberOfKnots - 1].Li = 0.; // the value will not be used, we define it for consistency

  // Set up the map (integer U) -> (knot index)

  int32_t* map = getUtoKnotMap();

  const int32_t iKnotMax = mNumberOfKnots - 2;

  //
  // With iKnotMax=nKnots-2 we map the U==Umax coordinate to the last [nKnots-2, nKnots-1] segment.
  // This trick allows one to avoid a special condition for this edge case.
  // Any U from [0,Umax] is mapped to some knot_i such, that the next knot_i+1 always exist
  //

  for (int32_t u = 0, iKnot = 0; u <= mUmax; u++) {
    if ((knotU[iKnot + 1] == u) && (iKnot < iKnotMax)) {
      iKnot = iKnot + 1;
    }
    map[u] = iKnot;
  }
}

template <class DataT, class FlatBase>
void Spline1DContainerBase<DataT, FlatBase>::print() const
{
  printf(" Spline 1D: \n");
  printf("  mNumberOfKnots = %d \n", mNumberOfKnots);
  printf("  mUmax = %d\n", mUmax);
  printf("  mUtoKnotMap = %p \n", (void*)mUtoKnotMap);
  printf("  knots: ");
  for (int32_t i = 0; i < mNumberOfKnots; i++) {
    printf("%d ", (int32_t)getKnot(i).u);
  }
  printf("\n");
}

#if !defined(GPUCA_STANDALONE)

template <class DataT, class FlatBase>
void Spline1DContainerBase<DataT, FlatBase>::approximateFunction(
  double xMin, double xMax,
  std::function<void(double x, double f[])> F,
  int32_t nAxiliaryDataPoints)
{
  /// approximate a function F with this spline
  if constexpr (std::is_same_v<FlatBase, FlatObject>) {
    Spline1DHelper<DataT> helper;
    helper.approximateFunction(*this, xMin, xMax, F, nAxiliaryDataPoints);
  }
}

template <class DataT, class FlatBase>
int32_t Spline1DContainerBase<DataT, FlatBase>::writeToFile(TFile& outf, const char* name)
{
  /// write a class object to the file
  if constexpr (std::is_same_v<FlatBase, FlatObject>) {
    return FlatObject::writeToFile(*this, outf, name);
  } else {
    return -1;
  }
}

template <class DataT, class FlatBase>
Spline1DContainerBase<DataT, FlatBase>* Spline1DContainerBase<DataT, FlatBase>::readFromFile(TFile& inpf, const char* name)
{
  /// read a class object from the file
  if constexpr (std::is_same_v<FlatBase, FlatObject>) {
    return FlatObject::readFromFile<Spline1DContainerBase<DataT, FlatBase>>(inpf, name);
  } else {
    return nullptr;
  }
}

#endif

template <class DataT, class FlatBase>
void Spline1DContainerBase<DataT, FlatBase>::cloneFromObject(const Spline1DContainerBase<DataT, FlatBase>& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description

  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatBase::cloneFromObject(obj, newFlatBufferPtr);
  mYdim = obj.mYdim;
  mNumberOfKnots = obj.mNumberOfKnots;
  mUmax = obj.mUmax;
  mXmin = obj.mXmin;
  mXtoUscale = obj.mXtoUscale;
  mUtoKnotMap = FlatBase::relocatePointer(oldFlatBufferPtr, this->mFlatBufferPtr, obj.mUtoKnotMap);
  mParameters = FlatBase::relocatePointer(oldFlatBufferPtr, this->mFlatBufferPtr, obj.mParameters);
}

template <class DataT, class FlatBase>
void Spline1DContainerBase<DataT, FlatBase>::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  char* oldFlatBufferPtr = this->mFlatBufferPtr;
  FlatBase::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = this->mFlatBufferPtr;
  this->mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}

template <class DataT, class FlatBase>
template <class OtherFlatBase>
void Spline1DContainerBase<DataT, FlatBase>::importFrom(const Spline1DContainerBase<DataT, OtherFlatBase>& src)
{
  /// Copy schema fields from a spline with a different FlatBase (e.g. FlatObject -> NoFlatObject).
  /// Pointers (mUtoKnotMap, mParameters) are set to nullptr; call setActualBufferAddress() afterward.
  mYdim = src.getYdimensions();
  mNumberOfKnots = src.getNumberOfKnots();
  mUmax = src.getUmax();
  mXmin = src.getXmin();
  mXtoUscale = src.getXtoUscale();
  this->mFlatBufferSize = src.getFlatBufferSize();
  mUtoKnotMap = nullptr;
  mParameters = nullptr;
}

template <class DataT, class FlatBase>
void Spline1DContainerBase<DataT, FlatBase>::destroy()
{
  /// See FlatObject for description
  mNumberOfKnots = 0;
  mUmax = 0;
  mYdim = 0;
  mXmin = 0.;
  mXtoUscale = 1.;
  mUtoKnotMap = nullptr;
  mParameters = nullptr;
  FlatBase::destroy();
}

template <class DataT, class FlatBase>
void Spline1DContainerBase<DataT, FlatBase>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description

  FlatBase::setActualBufferAddress(actualFlatBufferPtr);

  const int32_t uToKnotMapOffset = mNumberOfKnots * sizeof(Knot);
  mUtoKnotMap = reinterpret_cast<int32_t*>(this->mFlatBufferPtr + uToKnotMapOffset);
  int32_t parametersOffset = uToKnotMapOffset + (mUmax + 1) * sizeof(int32_t);
  if (mYdim > 0) {
    parametersOffset = this->alignSize(parametersOffset, getParameterAlignmentBytes());
  }
  mParameters = reinterpret_cast<DataT*>(this->mFlatBufferPtr + parametersOffset);
}

template <class DataT, class FlatBase>
void Spline1DContainerBase<DataT, FlatBase>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  mUtoKnotMap = FlatBase::relocatePointer(this->mFlatBufferPtr, futureFlatBufferPtr, mUtoKnotMap);
  mParameters = FlatBase::relocatePointer(this->mFlatBufferPtr, futureFlatBufferPtr, mParameters);
  FlatBase::setFutureBufferAddress(futureFlatBufferPtr);
}

#if !defined(GPUCA_STANDALONE)
template <class DataT, class FlatBase>
int32_t Spline1DContainerBase<DataT, FlatBase>::test(const bool draw, const bool drawDataPoints)
{
  return Spline1DHelper<DataT>::test(draw, drawDataPoints);
}
#endif // GPUCA_STANDALONE

template class o2::gpu::Spline1DContainerBase<float>;
template class o2::gpu::Spline1DContainerBase<double>;
template class o2::gpu::Spline1DContainer<float>;
template class o2::gpu::Spline1DContainer<double>;
template class o2::gpu::Spline1DSpec<float, 0, 2>;
template class o2::gpu::Spline1DSpec<double, 0, 2>;

// Explicit instantiations for NoFlatObject (used by TPCFastTransformPOD)
template class o2::gpu::Spline1DContainerBase<float, o2::gpu::NoFlatObject>;
template class o2::gpu::Spline1DContainerBase<double, o2::gpu::NoFlatObject>;
template class o2::gpu::Spline1DContainer<float, o2::gpu::NoFlatObject>;
template class o2::gpu::Spline1DContainer<double, o2::gpu::NoFlatObject>;
// importFrom instantiation for the FlatObject -> NoFlatObject conversion used in create()
template void o2::gpu::Spline1DContainerBase<float, o2::gpu::NoFlatObject>::importFrom<o2::gpu::FlatObject>(const o2::gpu::Spline1DContainerBase<float, o2::gpu::FlatObject>&);
template void o2::gpu::Spline1DContainerBase<double, o2::gpu::NoFlatObject>::importFrom<o2::gpu::FlatObject>(const o2::gpu::Spline1DContainerBase<double, o2::gpu::FlatObject>&);
