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

/// \file MultivariatePolynomial.h
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_MULTIVARIATEPOLYNOMIAL
#define ALICEO2_TPC_MULTIVARIATEPOLYNOMIAL

#include "GPUCommonDef.h"
#include "GPUCommonLogger.h"
#include "FlatObject.h"
#include "MultivariatePolynomialHelper.h"

#if !defined(GPUCA_GPUCODE)
#include <algorithm>
#include <type_traits>
#if !defined(GPUCA_STANDALONE)
#include <TFile.h>
#endif
#endif

namespace GPUCA_NAMESPACE::gpu
{

/// Class for multivariate polynomials.
/// The parameters of the coefficients have to be provided as input and can be obtained from TLinear fitter or from sklearn (PolynomialFeatures) etc.
/// The evaluation of the polynomials can be speed up by providing the dimensions and degree during compile time!
///
/// Usage: see example in testMultivarPolynomials.cxx
///    Dim > 0 && Degree > 0 : the number of dimensions and the degree is known at compile time
///    Dim = 0 && Degree = 0 : the number of dimensions and the degree will be set during runtime
template <unsigned int Dim, unsigned int Degree>
class MultivariatePolynomial : public FlatObject, public MultivariatePolynomialHelper<Dim, Degree>
{
 public:
#if !defined(GPUCA_GPUCODE)
  /// constructor for runtime evaluation of polynomial formula
  /// \param nDim number of dimensions
  /// \param degree degree of the polynomial
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (Dim == 0 && Degree == 0)), int>::type = 0>
  MultivariatePolynomial(const unsigned int nDim, const unsigned int degree) : MultivariatePolynomialHelper<Dim, Degree>{nDim, degree}, mNParams{this->getNParameters(degree, nDim)}
  {
    construct();
  }

  /// constructor for compile time evaluation of polynomial formula
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (Dim != 0 && Degree != 0)), int>::type = 0>
  MultivariatePolynomial() : mNParams{this->getNParameters(Degree, Dim)}
  {
    construct();
  }
#else
  /// default constructor
  MultivariatePolynomial() CON_DEFAULT;
#endif

  /// default destructor
  ~MultivariatePolynomial() CON_DEFAULT;

  /// Copy constructor
  MultivariatePolynomial(const MultivariatePolynomial& obj) { this->cloneFromObject(obj, nullptr); }

  /// ========== FlatObject functionality, see FlatObject class for description  =================
#if !defined(GPUCA_GPUCODE)
  /// cloning a container object (use newFlatBufferPtr=nullptr for simple copy)
  void cloneFromObject(const MultivariatePolynomial& obj, char* newFlatBufferPtr);

  /// move flat buffer to new location
  /// \param newBufferPtr new buffer location
  void moveBufferTo(char* newBufferPtr);
#endif

  /// destroy the object (release internal flat buffer)
  void destroy();

  /// set location of external flat buffer
  void setActualBufferAddress(char* actualFlatBufferPtr);

  /// set future location of the flat buffer
  void setFutureBufferAddress(char* futureFlatBufferPtr);
  /// ================================================================================================

  /// evaluates the polynomial for given coordinates
  /// \param x query coordinates
  GPUd() float eval(const float x[/*Dim*/]) const { return this->evalPol(mParams, x); }

#if !defined(GPUCA_GPUCODE)
  /// \return returns number of parameters of the polynomials
  unsigned int getNParams() const { return mNParams; }

  /// set the parameters for the coefficients of the polynomial
  /// \param params parameter for the coefficients
  void setParams(const float params[/*mNParams*/]) { std::copy(params, params + mNParams, mParams); }

  /// \param parameter which will be set
  /// \val value of the parameter
  void setParam(const unsigned int param, const float val) { mParams[param] = val; };

  /// \return returns the paramaters of the coefficients
  const float* getParams() const { return mParams; }

#ifndef GPUCA_STANDALONE
  /// load parameters from input file (which were written using the writeToFile method)
  /// \param inpf input file
  /// \parma name name of the object in the file
  void loadFromFile(TFile& inpf, const char* name);

  /// write parameters to file
  /// \param outf output file
  /// \param name name of the output object
  void writeToFile(TFile& outf, const char* name);
#endif

  /// converts the parameters to a container which can be written to a root file
  MultivariatePolynomialContainer getContainer() const { return MultivariatePolynomialContainer{this->getDim(), this->getDegree(), mNParams, mParams}; }

  /// set the parameters from MultivariatePolynomialContainer
  /// \param container container for the parameters
  void setFromContainer(const MultivariatePolynomialContainer& container);
#endif

 private:
  using DataTParams = float;     ///< data type of the parameters of the polynomials
  unsigned int mNParams{};       ///< number of parameters of the polynomial
  DataTParams* mParams{nullptr}; ///< parameters of the coefficients of the polynomial

#if !defined(GPUCA_GPUCODE)
  /// \return returns the size of the parameters
  std::size_t sizeOfParameters() const { return mNParams * sizeof(DataTParams); }

  // construct the object (flatbuffer)
  void construct();
#endif
};

//=================================================================================
//============================ inline implementations =============================
//=================================================================================

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
template <unsigned int Dim, unsigned int Degree>
void MultivariatePolynomial<Dim, Degree>::loadFromFile(TFile& inpf, const char* name)
{
  MultivariatePolynomialContainer* polTmp = nullptr;
  inpf.GetObject(name, polTmp);
  if (polTmp) {
    setFromContainer(*polTmp);
    delete polTmp;
  } else {
#ifndef GPUCA_ALIROOT_LIB
    LOGP(info, fmt::format("couldnt load object {} from input file", name));
#endif
  }
}

template <unsigned int Dim, unsigned int Degree>
void MultivariatePolynomial<Dim, Degree>::setFromContainer(const MultivariatePolynomialContainer& container)
{
  if constexpr (Dim > 0 && Degree > 0) {
    if (this->getDim() != container.mDim) {
#ifndef GPUCA_ALIROOT_LIB
      LOGP(info, fmt::format("wrong number of dimensions! this {} container {}", this->getDim(), container.mDim));
#endif
      return;
    }
    if (this->getDegree() != container.mDegree) {
#ifndef GPUCA_ALIROOT_LIB
      LOGP(info, fmt::format("wrong number of degrees! this {} container {}", this->getDegree(), container.mDegree));
#endif
      return;
    }
    setParams(container.mParams.data());
  } else {
    MultivariatePolynomial polTmp(container.mDim, container.mDegree);
    polTmp.setParams(container.mParams.data());
    this->cloneFromObject(polTmp, nullptr);
  }
}

template <unsigned int Dim, unsigned int Degree>
void MultivariatePolynomial<Dim, Degree>::writeToFile(TFile& outf, const char* name)
{
  const MultivariatePolynomialContainer cont = getContainer();
  outf.WriteObject(&cont, name);
}

template <unsigned int Dim, unsigned int Degree>
void MultivariatePolynomial<Dim, Degree>::cloneFromObject(const MultivariatePolynomial<Dim, Degree>& obj, char* newFlatBufferPtr)
{
  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  mNParams = obj.mNParams;
  if constexpr (Dim == 0 && Degree == 0) {
    this->mDim = obj.mDim;
    this->mDegree = obj.mDegree;
  }
  if (obj.mParams) {
    mParams = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mParams);
  }
}

template <unsigned int Dim, unsigned int Degree>
void MultivariatePolynomial<Dim, Degree>::moveBufferTo(char* newFlatBufferPtr)
{
  char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = mFlatBufferPtr;
  mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}

template <unsigned int Dim, unsigned int Degree>
void MultivariatePolynomial<Dim, Degree>::construct()
{
  FlatObject::startConstruction();
  const std::size_t flatbufferSize = sizeOfParameters();
  FlatObject::finishConstruction(flatbufferSize);
  mParams = reinterpret_cast<float*>(mFlatBufferPtr);
}
#endif

template <unsigned int Dim, unsigned int Degree>
void MultivariatePolynomial<Dim, Degree>::destroy()
{
  mParams = nullptr;
  FlatObject::destroy();
}

template <unsigned int Dim, unsigned int Degree>
void MultivariatePolynomial<Dim, Degree>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  mParams = reinterpret_cast<float*>(mFlatBufferPtr);
}

template <unsigned int Dim, unsigned int Degree>
void MultivariatePolynomial<Dim, Degree>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  mParams = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mParams);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

} // namespace GPUCA_NAMESPACE::gpu

#endif
