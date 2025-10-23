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

/// \file NDPiecewisePolynomials.inc
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_NDPIECEWISEPOLYNOMIALS_INC
#define ALICEO2_TPC_NDPIECEWISEPOLYNOMIALS_INC

#include <TLinearFitter.h>
#include <TFile.h>
#include "CommonUtils/TreeStreamRedirector.h"
#include "NDPiecewisePolynomials.h"

namespace o2::gpu
{

template <uint32_t Dim, uint32_t Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::dumpToTree(const uint32_t nSamplingPoints[/* Dim */], const char* outName, const char* treeName, const bool recreateFile) const
{
  o2::utils::TreeStreamRedirector pcstream(outName, recreateFile ? "RECREATE" : "UPDATE");

  double factor[Dim]{};
  for (uint32_t iDim = 0; iDim < Dim; ++iDim) {
    factor[iDim] = (mMax[iDim] - mMin[iDim]) / (nSamplingPoints[iDim] - 1);
  }

  std::vector<float> x(Dim);
  std::vector<uint32_t> ix(Dim);
  int32_t pos[Dim + 1]{0};

  for (;;) {
    checkPos(nSamplingPoints, pos);

    if (pos[Dim] == 1) {
      break;
    }

    for (uint32_t iDim = 0; iDim < Dim; ++iDim) {
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

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

template <uint32_t Dim, uint32_t Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::loadFromFile(TFile& inpf, const char* name)
{
  NDPiecewisePolynomialContainer* gridTmp = nullptr;
  inpf.GetObject(name, gridTmp);
  if (gridTmp) {
    setFromContainer(*gridTmp);
    delete gridTmp;
  } else {
    LOGP(info, "couldnt load object {} from input file", name);
  }
}

template <uint32_t Dim, uint32_t Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::loadFromFile(const char* fileName, const char* name)
{
  TFile f(fileName, "READ");
  loadFromFile(f, name);
}

template <uint32_t Dim, uint32_t Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::writeToFile(TFile& outf, const char* name) const
{
  const NDPiecewisePolynomialContainer cont = getContainer();
  outf.WriteObject(&cont, name);
}

template <uint32_t Dim, uint32_t Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::performFits(const std::function<double(const double x[/* Dim */])>& func, const uint32_t nAuxiliaryPoints[/* Dim */])
{
  const int32_t nTotalFits = getNPolynomials();
  LOGP(info, "Perform fitting of {}D-Polynomials of degree {} for a total of {} fits.", Dim, Degree, nTotalFits);

  MultivariatePolynomialHelper<0, 0, false> pol(Dim, Degree, InteractionOnly);
  TLinearFitter fitter = pol.getTLinearFitter();

  uint32_t nPoints = 1;
  for (uint32_t i = 0; i < Dim; ++i) {
    nPoints *= nAuxiliaryPoints[i];
  }

  std::vector<double> xCords;
  std::vector<double> response;
  xCords.reserve(Dim * nPoints);
  response.reserve(nPoints);

  uint32_t nPolynomials[Dim]{0};
  for (uint32_t i = 0; i < Dim; ++i) {
    nPolynomials[i] = getNPolynomials(i);
  }

  int32_t pos[Dim + 1]{0};
  uint32_t counter = 0;
  const int32_t printDebugForNFits = int32_t(nTotalFits / 20) + 1;

  for (;;) {
    const bool debug = !(++counter % printDebugForNFits);
    if (debug) {
      LOGP(info, "Performing fit {} out of {}", counter, nTotalFits);
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

template <uint32_t Dim, uint32_t Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::performFits(const std::vector<float>& x, const std::vector<float>& y)
{
  const int32_t nTotalFits = getNPolynomials();
  LOGP(info, "Perform fitting of {}D-Polynomials of degree {} for a total of {} fits.", Dim, Degree, nTotalFits);

  // approximate number of points
  uint32_t nPoints = 2 * y.size() / nTotalFits;

  // polynomial index -> indices to datapoints
  std::unordered_map<int32_t, std::vector<size_t>> dataPointsIndices;
  for (int32_t i = 0; i < nTotalFits; ++i) {
    dataPointsIndices[i].reserve(nPoints);
  }

  // check for each data point which polynomial to use
  for (size_t i = 0; i < y.size(); ++i) {
    std::array<int32_t, Dim> index;
    float xVal[Dim];
    std::copy(x.begin() + i * Dim, x.begin() + i * Dim + Dim, xVal);
    setIndex<Dim - 1>(xVal, index.data());

    std::array<int32_t, Dim> indexClamped{index};
    clamp<Dim - 1>(xVal, indexClamped.data());

    // check if data points are in the grid
    if (index == indexClamped) {
      // index of the polyniomial
      const uint32_t idx = getDataIndex(index.data()) / MultivariatePolynomialParametersHelper::getNParameters<Degree, Dim, InteractionOnly>();

      // store index to data point
      dataPointsIndices[idx].emplace_back(i);
    }
  }

  // for fitting
  MultivariatePolynomialHelper<0, 0, false> pol(Dim, Degree, InteractionOnly);
  TLinearFitter fitter = pol.getTLinearFitter();

  uint32_t counter = 0;
  const int32_t printDebugForNFits = int32_t(nTotalFits / 20) + 1;

  // temp storage for x and y values for fitting
  std::vector<double> xCords;
  std::vector<double> response;

  for (int32_t i = 0; i < nTotalFits; ++i) {
    const bool debug = !(++counter % printDebugForNFits);
    if (debug) {
      LOGP(info, "Performing fit {} out of {}", counter, nTotalFits);
    }

    // store values for fitting
    if (dataPointsIndices[i].empty()) {
      LOGP(info, "No data points to fit");
      continue;
    }

    const auto nP = dataPointsIndices[i].size();
    xCords.reserve(Dim * nP);
    response.reserve(nP);
    xCords.clear();
    response.clear();

    // add datapoints to fit
    for (size_t j = 0; j < nP; ++j) {
      const size_t idxOrig = dataPointsIndices[i][j];

      // insert x values at the end of xCords
      const int32_t idxXStart = idxOrig * Dim;
      xCords.insert(xCords.end(), x.begin() + idxXStart, x.begin() + idxXStart + Dim);
      response.emplace_back(y[idxOrig]);
    }

    // perform the fit on the points TODO make errors configurable
    std::vector<double> error;
    const auto params = MultivariatePolynomialHelper<0, 0, false>::fit(fitter, xCords, response, error, true);

    // store parameters
    std::copy(params.begin(), params.end(), &mParams[i * MultivariatePolynomialParametersHelper::getNParameters<Degree, Dim, InteractionOnly>()]);
  }
}

template <uint32_t Dim, uint32_t Degree, bool InteractionOnly>
void NDPiecewisePolynomials<Dim, Degree, InteractionOnly>::fitInnerGrid(const std::function<double(const double x[/* Dim */])>& func, const uint32_t nAuxiliaryPoints[/* Dim */], const int32_t currentIndex[/* Dim */], TLinearFitter& fitter, std::vector<double>& xCords, std::vector<double>& response)
{
  int32_t pos[Dim + 1]{0};

  // add points which will be used for the fit
  for (;;) {
    checkPos(nAuxiliaryPoints, pos);

    if (pos[Dim] == 1) {
      break;
    }

    for (uint32_t iDim = 0; iDim < Dim; ++iDim) {
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
  const uint32_t index = getDataIndex(currentIndex);
  std::copy(params.begin(), params.end(), &mParams[index]);
}

} // namespace o2::gpu

#endif // !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

#endif // ALICEO2_TPC_NDPIECEWISEPOLYNOMIALS_INC
