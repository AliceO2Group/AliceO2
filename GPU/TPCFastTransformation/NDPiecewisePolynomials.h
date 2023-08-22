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

/// \file NDPiecewisePolynomials.h
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_NDPIECEWISEPOLYNOMIALS
#define ALICEO2_TPC_NDPIECEWISEPOLYNOMIALS

#include "GPUCommonLogger.h"
#include "FlatObject.h"
#include "MultivariatePolynomialHelper.h"
#include "GPUCommonMath.h"

#if !defined(GPUCA_GPUCODE)
#include <vector>
#if !defined(GPUCA_STANDALONE)
#include "TLinearFitter.h"
#ifndef GPUCA_ALIROOT_LIB
#include "CommonUtils/TreeStreamRedirector.h"
#endif
#include <TFile.h>
#endif
#endif

namespace GPUCA_NAMESPACE::gpu
{

#if !defined(GPUCA_GPUCODE)
/// simple struct to enable writing the NDPiecewisePolynomials to file
struct NDPiecewisePolynomialContainer {

  /// constructor
  /// \param dim number of dimensions of the polynomial
  /// \param degree degree of the polynomials
  /// \param nParameters number of parameters
  /// \param params parmaeters
  /// \param interactionOnly consider only interaction terms
  /// \param xmin minimum coordinates of the grid
  /// \param xmax maximum coordinates of the grid
  /// \param nVertices number of vertices: defines number of fits per dimension
  NDPiecewisePolynomialContainer(const unsigned int dim, const unsigned int degree, const unsigned int nParameters, const float params[/* nParameters*/], const bool interactionOnly, const float xMin[/* Dim*/], const float xMax[/* Dim*/], const unsigned int nVertices[/* Dim*/]) : mDim{dim}, mDegree{degree}, mParams{params, params + nParameters}, mInteractionOnly{interactionOnly}, mMin{xMin, xMin + dim}, mMax{xMax, xMax + dim}, mN{nVertices, nVertices + dim} {};

  /// for ROOT I/O
  NDPiecewisePolynomialContainer() = default;

  const unsigned int mDim{};            ///< number of dimensions of the polynomial
  const unsigned int mDegree{};         ///< degree of the polynomials
  const std::vector<float> mParams{};   ///< parameters of the polynomial
  const bool mInteractionOnly{};        ///< consider only interaction terms
  const std::vector<float> mMin{};      ///< min vertices positions of the grid
  const std::vector<float> mMax{};      ///< max vertices positions of the grid
  const std::vector<unsigned int> mN{}; ///< number of vertices for each dimension
};
#endif

/// class for piecewise polynomial fits on a regular grid
/// This class can be used to perform independent fits of nth-degree polynomials on a regular grid.
/// The fitting should be performed on CPUs. The evaluation can be performed on CPU or GPU.
/// A smooth function which can be evaluated on the full grid is expected to be provided when performing the fits.
/// The main difference to splines is that the performed polynomial fits have no restrictions on the boundaries inside of the grid (it is not guaranteed that the resulting fits will be smooth at the grid vertices)
/// The advantage to splines is the faster evaluation of the polynomials, which are still reasonable to use in higher dimensions, when performance is critical.
///
/// For usage see: testMultivarPolynomials.cxx
///
/// TODO: add possibillity to perform the fits on scattered data points (+add weighting of points)
///
/// \tparam Dim number of dimensions
/// \tparam Degree degree of the polynomials
/// \tparam InteractionOnly: consider only interaction terms: ignore x[0]*x[0]..., x[1]*x[1]*x[2]... etc. terms (same feature as 'interaction_only' in sklearn https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
///                          can be used for N-linear interpolation (https://en.wikipedia.org/wiki/Trilinear_interpolation#Alternative_algorithm)
template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
class NDPiecewisePolynomials : public FlatObject
{
 public:
#ifndef GPUCA_GPUCODE
  /// constructor
  /// \param min minimum coordinates of the grid
  /// \param max maximum coordinates of the grid (note: the resulting polynomials can NOT be evaluated at the maximum coordinates: only at min <= X < max)
  /// \param n number of vertices: defines number of fits per dimension: nFits = n - 1. n should be at least 2 to perform one fit
  NDPiecewisePolynomials(const float min[/* Dim */], const float max[/* Dim */], const unsigned int n[/* Dim */]) { init(min, max, n); }
#endif
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// constructor construct and object by initializing it from an object stored in a Root file
  /// \param fileName name of the file
  /// \param name name of the object
  NDPiecewisePolynomials(const char* fileName, const char* name)
  {
    TFile f(fileName, "READ");
    loadFromFile(f, name);
  };
#endif
  /// default constructor
  NDPiecewisePolynomials() CON_DEFAULT;

  /// default destructor
  ~NDPiecewisePolynomials() CON_DEFAULT;

  /// Copy constructor
  NDPiecewisePolynomials(const NDPiecewisePolynomials& obj) { cloneFromObject(obj, nullptr); }

  /// ========== FlatObject functionality, see FlatObject class for description  =================
#if !defined(GPUCA_GPUCODE)
  /// cloning a container object (use newFlatBufferPtr=nullptr for simple copy)
  void cloneFromObject(const NDPiecewisePolynomials& obj, char* newFlatBufferPtr);

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

  /// evalute at given coordinate with boundary check (note: input values will be changed/clamped to the grid)
  /// \param x coordinates where to interpolate
  GPUd() float eval(float x[/* Dim */]) const
  {
    int index[Dim];
    setIndex<Dim - 1>(x, index);
    clamp<Dim - 1>(x, index);
    return MultivariatePolynomialHelper<Dim, Degree, InteractionOnly>::evalPol(getParameters(index), x);
  }

  /// evalute at given coordinate (note: A boundary check for the provided coordinates is not performed!)
  /// \param x coordinates where to interpolate (note: the resulting polynomials can NOT be evaluated at the maximum coordinates: only at min <= X < max)
  GPUd() float evalUnsafe(const float x[/* Dim */]) const
  {
    int index[Dim];
    setIndex<Dim - 1>(x, index);
    return MultivariatePolynomialHelper<Dim, Degree, InteractionOnly>::evalPol(getParameters(index), x);
  }

  /// evaluate specific polynomial at given index for given coordinate
  /// \param x coordinates where to interpolate
  /// \param index index of the polynomial
  GPUd() float evalPol(const float x[/* Dim */], const int index[/* Dim */]) const { return MultivariatePolynomialHelper<Dim, Degree, InteractionOnly>::evalPol(getParameters(index), x); }

  /// \return returns min range for given dimension
  GPUd() float getXMin(const unsigned int dim) const { return mMin[dim]; }

  /// \return returns max range for given dimension
  GPUd() float getXMax(const unsigned int dim) const { return mMax[dim]; }

  /// \return returns inverse spacing for given dimension
  GPUd() float getInvSpacing(const unsigned int dim) const { return mInvSpacing[dim]; }

  /// \return returns number of vertices for given dimension
  GPUd() unsigned int getNVertices(const unsigned int dim) const { return mN[dim]; }

  /// \return returns number of polynomial fits for given dimension
  GPUd() unsigned int getNPolynomials(const unsigned int dim) const { return mN[dim] - 1; }

  /// \return returns the parameters of the coefficients
  GPUd() const float* getParams() const { return mParams; }

#if !defined(GPUCA_GPUCODE)
  /// Setting directly the parameters of the polynomials
  void setParams(const float params[/* getNParameters() */]) { std::copy(params, params + getNParameters(), mParams); }

  /// initalize the members
  /// \param min minimum coordinates of the grid
  /// \param max maximum coordinates of the grid (note: the resulting polynomials can NOT be evaluated at the maximum coordinates: only at min <= X < max)
  /// \param n number of vertices: defines number of fits per dimension: nFits = n - 1. n should be at least 2 to perform one fit
  void init(const float min[/* Dim */], const float max[/* Dim */], const unsigned int n[/* Dim */]);

#ifndef GPUCA_STANDALONE
  /// perform the polynomial fits on the grid
  /// \param func function which returns for every input x on the defined grid the true value
  /// \param nAuxiliaryPoints number of points which will be used for the fits (should be at least 2)
  void performFits(const std::function<double(const double x[/* Dim */])>& func, const unsigned int nAuxiliaryPoints[/* Dim */]);

  /// load parameters from input file (which were written using the writeToFile method)
  /// \param inpf input file
  /// \param name name of the object in the file
  void loadFromFile(TFile& inpf, const char* name);

  /// write parameters to file
  /// \param outf output file
  /// \param name name of the output object
  void writeToFile(TFile& outf, const char* name) const;

  /// setting default polynomials which just returns 1
  void setDefault();

  /// dump the polynomials to tree for visualisation
  /// \param nSamplingPoints number of sampling points per dimension
  /// \param outName name of the output file
  /// \param treeName name of the tree
  /// \param recreateFile create new output file or update the output file
  void dumpToTree(const unsigned int nSamplingPoints[/* Dim */], const char* outName = "debug.root", const char* treeName = "tree", const bool recreateFile = true) const;

  /// \return returns total number of polynomial fits
  unsigned int getNPolynomials() const;
#endif

  /// converts the class to a container which can be written to a root file
  NDPiecewisePolynomialContainer getContainer() const { return NDPiecewisePolynomialContainer{Dim, Degree, getNParameters(), mParams, InteractionOnly, mMin, mMax, mN}; }

  /// set the parameters from NDPiecewisePolynomialContainer
  /// \param container container for the parameters
  void setFromContainer(const NDPiecewisePolynomialContainer& container);

  /// \return returns the total number of stored parameters
  unsigned int getNParameters() const { return getNPolynomials() * MultivariatePolynomialParametersHelper::getNParameters(Degree, Dim, InteractionOnly); }
#endif

  /// \return returns number of dimensions of the polynomials
  GPUd() static constexpr unsigned int getDim() { return Dim; }

  /// \return returns the degree of the polynomials
  GPUd() static constexpr unsigned int getDegree() { return Degree; }

  /// \return returns whether only interaction terms are considered
  GPUd() static constexpr bool isInteractionOnly() { return InteractionOnly; }

 private:
  using DataTParams = float;     ///< data type of the parameters of the polynomials
  float mMin[Dim]{};             ///< min vertices positions of the grid
  float mMax[Dim]{};             ///< max vertices positions of the grid
  float mInvSpacing[Dim]{};      ///< inverse spacings of the grid
  unsigned int mN[Dim]{};        ///< number of vertices for each dimension
  DataTParams* mParams{nullptr}; ///< storage for the parameters

  /// \returns vertex number for given position and dimension
  /// \param x position
  /// \param dim dimension
  GPUd() int getVertex(const float x, const unsigned int dim) const { return ((x - mMin[dim]) * mInvSpacing[dim]); }

  /// returns terms which are needed to calculate the index for the grid for given dimension
  /// \param dim dimension
  GPUd() unsigned int getTerms(const unsigned int dim) const { return (dim == 0) ? 1 : (mN[dim - 1] - 1) * getTerms(dim - 1); }

  /// returns index for accessing the parameter on the grid
  /// \param ix index per dimension
  GPUd() unsigned int getDataIndex(const int ix[/* Dim */]) const { return getDataIndex<Dim - 1>(ix) * MultivariatePolynomialParametersHelper::getNParameters(Degree, Dim, InteractionOnly); }

  /// helper function to get the index
  template <unsigned int DimTmp>
  GPUd() unsigned int getDataIndex(const int ix[/* Dim */]) const;

  /// \return returns pointer to memory where the parameters for given indices are stored
  /// \param index indices of polynomials
  GPUd() const float* getParameters(const int index[/* Dim */]) const { return &mParams[getDataIndex(index)]; }

  /// helper function to fill array containing the indices to where the parameters are stored
  template <unsigned int DimTmp>
  GPUd() void setIndex(const float x[/* Dim */], int index[/* Dim */]) const;

  /// clamp input coordinates to avoid extrapolation
  template <unsigned int DimTmp>
  GPUd() void clamp(float x[/* Dim */], int index[/* Dim */]) const;

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// perform the actual fit in the grid
  /// \param func function which returns for evey input x the true value
  /// \param nAuxiliaryPoints number of points which will be used for the fits (should be at least 2)
  /// \param currentIndex to keep track of where (indices on the grid) the fit is beeing performed
  /// \param fitter TLinearFitter which is used for performing the fits
  /// \param xCords buffer for x-coordinates
  /// \param response buffer for y-coordinates
  void fitInnerGrid(const std::function<double(const double x[/* Dim */])>& func, const unsigned int nAuxiliaryPoints[/* Dim */], const int currentIndex[/* Dim */], TLinearFitter& fitter, std::vector<double>& xCords, std::vector<double>& response);

  /// heler function to loop over all dimensions
  void checkPos(const unsigned int iMax[/* Dim */], int pos[/* Dim */]) const;

  /// \return returns step width of the inner grid
  /// \param dim dimension
  /// \param nAuxiliaryPoints number of Auxiliary points for given dimension
  double getStepWidth(const unsigned int dim, const int nAuxiliaryPoints) const { return 1 / (static_cast<double>(mInvSpacing[dim]) * (nAuxiliaryPoints - 1)); }

  /// \return returns vertex position for given index and dimension
  /// \param ix index
  /// \param dim dimension
  double getVertexPosition(const unsigned int ix, const int dim) const { return ix / static_cast<double>(mInvSpacing[dim]) + mMin[dim]; }
#endif

#if !defined(GPUCA_GPUCODE)
  /// \return returns the size of the parameters
  std::size_t sizeOfParameters() const { return getNParameters() * sizeof(DataTParams); }

  // construct the object (flatbuffer)
  void construct();
#endif

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(NDPiecewisePolynomials, 1);
#endif
};

//=================================================================================
//============================ inline implementations =============================
//=================================================================================

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::loadFromFile(TFile& inpf, const char* name)
{
  NDPiecewisePolynomialContainer* gridTmp = nullptr;
  inpf.GetObject(name, gridTmp);
  if (gridTmp) {
    setFromContainer(*gridTmp);
    delete gridTmp;
  } else {
#ifndef GPUCA_ALIROOT_LIB
    LOGP(info, fmt::format("couldnt load object {} from input file", name));
#endif
  }
}
template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::setFromContainer(const NDPiecewisePolynomialContainer& container)
{
  if (Dim != container.mDim) {
#ifndef GPUCA_ALIROOT_LIB
    LOGP(info, fmt::format("wrong number of dimensions! this {} container {}", Dim, container.mDim));
#endif
    return;
  }
  if (Degree != container.mDegree) {
#ifndef GPUCA_ALIROOT_LIB
    LOGP(info, fmt::format("wrong number of degrees! this {} container {}", Degree, container.mDegree));
#endif
    return;
  }
  if (InteractionOnly != container.mInteractionOnly) {
#ifndef GPUCA_ALIROOT_LIB
    LOGP(info, fmt::format("InteractionOnly is set for this object to {}, but stored as {} in the container", InteractionOnly, container.mInteractionOnly));
#endif
    return;
  }
  init(container.mMin.data(), container.mMax.data(), container.mN.data());
  setParams(container.mParams.data());
}
template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::writeToFile(TFile& outf, const char* name) const
{
  const NDPiecewisePolynomialContainer cont = getContainer();
  outf.WriteObject(&cont, name);
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::setDefault()
{
  const auto nParamsPerPol = MultivariatePolynomialParametersHelper::getNParameters(Degree, Dim, InteractionOnly);
  const auto nPols = getNPolynomials();
  std::vector<float> params(nParamsPerPol);
  params.front() = 1;
  for (auto i = 0; i < nPols; ++i) {
    std::copy(params.begin(), params.end(), &mParams[i * nParamsPerPol]);
  }
}
#endif

#ifndef GPUCA_GPUCODE
template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::cloneFromObject(const NDPiecewisePolynomials<Dim, Degree, InteractionOnly>& obj, char* newFlatBufferPtr)
{
  const char* oldFlatBufferPtr = obj.mFlatBufferPtr;
  FlatObject::cloneFromObject(obj, newFlatBufferPtr);
  for (unsigned int i = 0; i < Dim; ++i) {
    mMin[i] = obj.mMin[i];
    mMax[i] = obj.mMax[i];
    mInvSpacing[i] = obj.mInvSpacing[i];
    mN[i] = obj.mN[i];
  }
  if (obj.mParams) {
    mParams = FlatObject::relocatePointer(oldFlatBufferPtr, mFlatBufferPtr, obj.mParams);
  }
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::moveBufferTo(char* newFlatBufferPtr)
{
  char* oldFlatBufferPtr = mFlatBufferPtr;
  FlatObject::moveBufferTo(newFlatBufferPtr);
  char* currFlatBufferPtr = mFlatBufferPtr;
  mFlatBufferPtr = oldFlatBufferPtr;
  setActualBufferAddress(currFlatBufferPtr);
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::construct()
{
  FlatObject::startConstruction();
  const std::size_t flatbufferSize = sizeOfParameters();
  FlatObject::finishConstruction(flatbufferSize);
  mParams = reinterpret_cast<DataTParams*>(mFlatBufferPtr);
}
#endif

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::destroy()
{
  mParams = nullptr;
  FlatObject::destroy();
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::setActualBufferAddress(char* actualFlatBufferPtr)
{
  FlatObject::setActualBufferAddress(actualFlatBufferPtr);
  mParams = reinterpret_cast<DataTParams*>(mFlatBufferPtr);
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  mParams = FlatObject::relocatePointer(mFlatBufferPtr, futureFlatBufferPtr, mParams);
  FlatObject::setFutureBufferAddress(futureFlatBufferPtr);
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
template <unsigned int DimTmp>
GPUdi() unsigned int NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::getDataIndex(const int ix[/* Dim */]) const
{
  if constexpr (DimTmp > 0) {
    return ix[DimTmp] * getTerms(DimTmp) + getDataIndex<DimTmp - 1>(ix);
  }
  return ix[DimTmp];
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
template <unsigned int DimTmp>
GPUdi() void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::setIndex(const float x[/* Dim */], int index[/* Dim */]) const
{
  index[DimTmp] = getVertex(x[DimTmp], DimTmp);
  if constexpr (DimTmp > 0) {
    return setIndex<DimTmp - 1>(x, index);
  }
  return;
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
template <unsigned int DimTmp>
GPUdi() void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::clamp(float x[/* Dim */], int index[/* Dim */]) const
{
  if (index[DimTmp] <= 0) {
    index[DimTmp] = 0;

    if (x[DimTmp] < mMin[DimTmp]) {
      x[DimTmp] = mMin[DimTmp];
    }

  } else {
    if (index[DimTmp] >= int(mN[DimTmp] - 1)) {
      index[DimTmp] = mN[DimTmp] - 2;
      x[DimTmp] = mMax[DimTmp];
    }
  }

  if constexpr (DimTmp > 0) {
    return clamp<DimTmp - 1>(x, index);
  }
}

#ifndef GPUCA_GPUCODE
template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::init(const float min[], const float max[], const unsigned int n[])
{
  for (unsigned int i = 0; i < Dim; ++i) {
    mMin[i] = min[i];
    mMax[i] = max[i];
    mN[i] = n[i];
    mInvSpacing[i] = (mN[i] - 1) / (mMax[i] - mMin[i]);
  }
  construct();
}
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
unsigned int NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::getNPolynomials() const
{
  unsigned int nP = getNPolynomials(0);
  for (unsigned int i = 1; i < Dim; ++i) {
    nP *= getNPolynomials(i);
  }
  return nP;
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::checkPos(const unsigned int iMax[/* Dim */], int pos[/* Dim */]) const
{
  for (unsigned int i = 0; i < Dim; ++i) {
    if (pos[i] == int(iMax[i])) {
      ++pos[i + 1];
      std::fill_n(pos, i + 1, 0);
    }
  }
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::performFits(const std::function<double(const double x[/* Dim */])>& func, const unsigned int nAuxiliaryPoints[/* Dim */])
{
  const int nTotalFits = getNPolynomials();
#ifndef GPUCA_ALIROOT_LIB
  LOGP(info, "Perform fitting of {}D-Polynomials of degree {} for a total of {} fits.", Dim, Degree, nTotalFits);
#endif

  MultivariatePolynomialHelper<0, 0, false> pol(Dim, Degree, InteractionOnly);
  TLinearFitter fitter = pol.getTLinearFitter();

  unsigned int nPoints = 1;
  for (unsigned int i = 0; i < Dim; ++i) {
    nPoints *= nAuxiliaryPoints[i];
  }

  std::vector<double> xCords;
  std::vector<double> response;
  xCords.reserve(Dim * nPoints);
  response.reserve(nPoints);

  unsigned int nPolynomials[Dim]{0};
  for (unsigned int i = 0; i < Dim; ++i) {
    nPolynomials[i] = getNPolynomials(i);
  }

  int pos[Dim + 1]{0};
  unsigned int counter = 0;
  const int printDebugForNFits = int(nTotalFits / 20) + 1;

  for (;;) {
    const bool debug = !(++counter % printDebugForNFits);
    if (debug) {
#ifndef GPUCA_ALIROOT_LIB
      LOGP(info, "Peforming fit {} out of {}", counter, nTotalFits);
#endif
    }

    checkPos(nPolynomials, pos);

    if (pos[Dim] == 1) {
      break;
    }

    xCords.clear();
    response.clear();
    fitInnerGrid(func, nAuxiliaryPoints, pos, fitter, xCords, response);
    ++pos[0];
  }
}

template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::fitInnerGrid(const std::function<double(const double x[/* Dim */])>& func, const unsigned int nAuxiliaryPoints[/* Dim */], const int currentIndex[/* Dim */], TLinearFitter& fitter, std::vector<double>& xCords, std::vector<double>& response)
{
  int pos[Dim + 1]{0};

  // add points which will be used for the fit
  for (;;) {
    checkPos(nAuxiliaryPoints, pos);

    if (pos[Dim] == 1) {
      break;
    }

    for (unsigned int iDim = 0; iDim < Dim; ++iDim) {
      const double stepWidth = getStepWidth(iDim, nAuxiliaryPoints[iDim]);
      const double vertexPos = getVertexPosition(currentIndex[iDim], iDim);
      const double realPosTmp = vertexPos + pos[iDim] * stepWidth;
      xCords.emplace_back(realPosTmp);
    }

    // get response for last added points
    const double responseTmp = func(&xCords[xCords.size() - Dim]);
    response.emplace_back(responseTmp);
    ++pos[0];
  }

  // perform the fit on the points TODO make errors configurable
  std::vector<double> error;
  const auto params = MultivariatePolynomialHelper<0, 0, false>::fit(fitter, xCords, response, error, true);

  // store parameters
  const unsigned int index = getDataIndex(currentIndex);
  std::copy(params.begin(), params.end(), &mParams[index]);
}

#ifndef GPUCA_ALIROOT_LIB
template <unsigned int Dim, unsigned int Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::dumpToTree(const unsigned int nSamplingPoints[/* Dim */], const char* outName, const char* treeName, const bool recreateFile) const
{
  o2::utils::TreeStreamRedirector pcstream(outName, recreateFile ? "RECREATE" : "UPDATE");

  double factor[Dim]{};
  for (unsigned int iDim = 0; iDim < Dim; ++iDim) {
    factor[iDim] = (mMax[iDim] - mMin[iDim]) / (nSamplingPoints[iDim] - 1);
  }

  std::vector<float> x(Dim);
  std::vector<unsigned int> ix(Dim);
  int pos[Dim + 1]{0};

  for (;;) {
    checkPos(nSamplingPoints, pos);

    if (pos[Dim] == 1) {
      break;
    }

    for (unsigned int iDim = 0; iDim < Dim; ++iDim) {
      ix[iDim] = pos[iDim];
      x[iDim] = mMin[iDim] + pos[iDim] * factor[iDim];
    }

    float value = eval(x.data());
    pcstream << treeName
             << "ix=" << ix
             << "x=" << x
             << "value=" << value
             << "\n";

    ++pos[0];
  }
  pcstream.Close();
}
#endif

#endif

} // namespace GPUCA_NAMESPACE::gpu

#endif
