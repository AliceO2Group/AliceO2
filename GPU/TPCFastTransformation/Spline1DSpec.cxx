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
#endif

using namespace std;
using namespace o2::gpu;

template <class DataT>
void Spline1DContainer<DataT, FlatObject>::recreate(int32_t nYdim, int32_t numberOfKnots)
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

template <class DataT>
void Spline1DContainer<DataT, FlatObject>::recreate(int32_t nYdim, int32_t numberOfKnots, const int32_t inputKnots[])
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

  this->mYdim = (nYdim >= 0) ? nYdim : 0;

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

  this->mNumberOfKnots = knotU.size();
  this->mUmax = knotU.back();
  this->mXmin = 0.;
  this->mXtoUscale = 1.;

  const int32_t uToKnotMapOffset = this->mNumberOfKnots * sizeof(Knot<DataT>);
  int32_t parametersOffset = uToKnotMapOffset + (this->mUmax + 1) * sizeof(int32_t);
  int32_t bufferSize = parametersOffset;
  if (this->mYdim > 0) {
    parametersOffset = this->alignSize(bufferSize, this->getParameterAlignmentBytes());
    bufferSize = parametersOffset + this->getSizeOfParameters();
  }

  FlatObject::finishConstruction(bufferSize);

  this->mUtoKnotMap = reinterpret_cast<int32_t*>(this->mFlatBufferPtr + uToKnotMapOffset);
  this->mParameters = reinterpret_cast<DataT*>(this->mFlatBufferPtr + parametersOffset);

  for (int32_t i = 0; i < this->getNumberOfParameters(); i++) {
    this->mParameters[i] = 0;
  }

  Knot<DataT>* s = getKnots();

  for (int32_t i = 0; i < this->mNumberOfKnots; i++) {
    s[i].u = knotU[i];
  }

  for (int32_t i = 0; i < this->mNumberOfKnots - 1; i++) {
    s[i].Li = 1. / (s[i + 1].u - s[i].u); // do division in double
  }

  s[this->mNumberOfKnots - 1].Li = 0.; // the value will not be used, we define it for consistency

  // Set up the map (integer U) -> (knot index)

  int32_t* map = getUtoKnotMap();

  const int32_t iKnotMax = this->mNumberOfKnots - 2;

  //
  // With iKnotMax=nKnots-2 we map the U==Umax coordinate to the last [nKnots-2, nKnots-1] segment.
  // This trick allows one to avoid a special condition for this edge case.
  // Any U from [0,Umax] is mapped to some knot_i such, that the next knot_i+1 always exist
  //

  for (int32_t u = 0, iKnot = 0; u <= this->mUmax; u++) {
    if ((knotU[iKnot + 1] == u) && (iKnot < iKnotMax)) {
      iKnot = iKnot + 1;
    }
    map[u] = iKnot;
  }
}

template <class DataT>
void Spline1DContainer<DataT, FlatObject>::print() const
{
  printf(" Spline 1D: \n");
  printf("  mNumberOfKnots = %d \n", this->mNumberOfKnots);
  printf("  mUmax = %d\n", this->mUmax);
  printf("  mUtoKnotMap = %p \n", (void*)this->mUtoKnotMap);
  printf("  knots: ");
  for (int32_t i = 0; i < this->mNumberOfKnots; i++) {
    printf("%d ", (int32_t)getKnot(i).u);
  }
  printf("\n");
}

#if !defined(GPUCA_STANDALONE)

template <class DataT>
void Spline1DContainer<DataT, FlatObject>::approximateFunction(
  double xMin, double xMax,
  std::function<void(double x, double f[])> F,
  int32_t nAxiliaryDataPoints)
{
  /// approximate a function F with this spline
  Spline1DHelper<DataT> helper;
  helper.approximateFunction(*this, xMin, xMax, F, nAxiliaryDataPoints);
}

template <class DataT>
int32_t Spline1DContainer<DataT, FlatObject>::writeToFile(TFile& outf, const char* name)
{
  /// write a class object to the file
  return FlatObject::writeToFile(*this, outf, name);
}

template <class DataT>
Spline1DContainer<DataT, FlatObject>* Spline1DContainer<DataT, FlatObject>::readFromFile(TFile& inpf, const char* name)
{
  /// read a class object from the file
  return FlatObject::readFromFile<Spline1DContainer<DataT, FlatObject>>(inpf, name);
}

template <class DataT>
int32_t Spline1DContainer<DataT, FlatObject>::test(const bool draw, const bool drawDataPoints)
{
  return Spline1DHelper<DataT>::test(draw, drawDataPoints);
}

#endif

template <class DataT>
void Spline1DContainer<DataT, FlatObject>::cloneFromObject(const Spline1DContainer<DataT, FlatObject>& obj, char* newFlatBufferPtr)
{
  /// See FlatObject for description
  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  this->mYdim = obj.mYdim;
  this->mNumberOfKnots = obj.mNumberOfKnots;
  this->mUmax = obj.mUmax;
  this->mXmin = obj.mXmin;
  this->mXtoUscale = obj.mXtoUscale;
  this->mUtoKnotMap = FlatObject::relocatePointer(oldFlatBufferPtr, this->mFlatBufferPtr, obj.mUtoKnotMap);
  this->mParameters = FlatObject::relocatePointer(oldFlatBufferPtr, this->mFlatBufferPtr, obj.mParameters);
}

template <class DataT>
void Spline1DContainer<DataT, FlatObject>::moveBufferTo(char* newFlatBufferPtr)
{
  /// See FlatObject for description
  char* oldFlatBufferPtr = this->mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = this->mFlatBufferPtr;
  this->mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}

template <class DataT>
template <class OtherFlatBase>
void Spline1DContainer<DataT, FlatObject>::importFrom(const Spline1DContainerBase<DataT, OtherFlatBase>& src)
{
  /// Copy schema fields from a spline with a different FlatBase (e.g. FlatObject -> NoFlatObject).
  /// Pointers (mUtoKnotMap, mParameters) are set to nullptr; call setActualBufferAddress() afterward.
  this->mYdim = src.getYdimensions();
  this->mNumberOfKnots = src.getNumberOfKnots();
  this->mUmax = src.getUmax();
  this->mXmin = src.getXmin();
  this->mXtoUscale = src.getXtoUscale();
  this->mFlatBufferSize = src.getFlatBufferSize();
  this->mUtoKnotMap = nullptr;
  this->mParameters = nullptr;
}

template <class DataT>
void Spline1DContainer<DataT, FlatObject>::destroy()
{
  this->mNumberOfKnots = 0;
  this->mUmax = 0;
  this->mYdim = 0;
  this->mXmin = 0.;
  this->mXtoUscale = 1.;
  this->mUtoKnotMap = nullptr;
  this->mParameters = nullptr;
  FlatObject::destroy();
}

template <class DataT>
void Spline1DContainer<DataT, FlatObject>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// See FlatObject for description
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);

  const int32_t uToKnotMapOffset = this->mNumberOfKnots * sizeof(Knot<DataT>);
  this->mUtoKnotMap = reinterpret_cast<int32_t*>(this->mFlatBufferPtr + uToKnotMapOffset);
  int32_t parametersOffset = uToKnotMapOffset + (this->mUmax + 1) * sizeof(int32_t);
  if (this->mYdim > 0) {
    parametersOffset = this->alignSize(parametersOffset, this->getParameterAlignmentBytes());
  }
  mParameters = reinterpret_cast<DataT*>(this->mFlatBufferPtr + parametersOffset);
}

template <class DataT>
void Spline1DContainer<DataT, FlatObject>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// See FlatObject for description
  this->mUtoKnotMap = FlatObject::relocatePointer(this->mFlatBufferPtr, futureFlatBufferPtr, this->mUtoKnotMap);
  this->mParameters = FlatObject::relocatePointer(this->mFlatBufferPtr, futureFlatBufferPtr, this->mParameters);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

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
template void o2::gpu::Spline1DContainer<float, o2::gpu::NoFlatObject>::importFrom<o2::gpu::FlatObject>(const o2::gpu::Spline1DContainerBase<float, o2::gpu::FlatObject>&);
template void o2::gpu::Spline1DContainer<double, o2::gpu::NoFlatObject>::importFrom<o2::gpu::FlatObject>(const o2::gpu::Spline1DContainerBase<double, o2::gpu::FlatObject>&);
// importFrom for FlatObject container (FlatObject -> FlatObject, e.g. when using as a copy tool)
template void o2::gpu::Spline1DContainer<float, o2::gpu::FlatObject>::importFrom<o2::gpu::FlatObject>(const o2::gpu::Spline1DContainerBase<float, o2::gpu::FlatObject>&);
