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

/// \file  MultivariatePolynomialHelper.cxx
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#include "MultivariatePolynomialHelper.h"
#include "GPUCommonLogger.h"

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
#include "TLinearFitter.h"
#include <algorithm>
#endif

using namespace GPUCA_NAMESPACE::gpu;

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
void MultivariatePolynomialHelper<0, 0, false>::print() const
{
#ifndef GPUCA_NO_FMT
  LOGP(info, getFormula().c_str());
#endif
}

std::string MultivariatePolynomialHelper<0, 0, false>::getTLinearFitterFormula() const
{
  std::string formula = getFormula();
  formula.replace(formula.find("p"), formula.find("]") + 1, "1");
  size_t pos = std::string::npos;
  while ((pos = formula.find("* p")) != std::string::npos) {
    size_t end_pos = formula.find("]", pos) + 2;
    formula.erase(pos, end_pos - pos);
  }

  while ((pos = formula.find(" + ")) != std::string::npos) {
    formula.replace(pos + 1, 1, "++");
  }
  return formula;
}

std::string MultivariatePolynomialHelper<0, 0, false>::getFormula() const
{
  std::string formula = "";
#ifndef GPUCA_NO_FMT
  const auto terms = getTerms();
  for (int i = 0; i < (int)terms.size() - 1; ++i) {
    formula += fmt::format("{} + ", terms[i]);
  }
  formula += terms.back();
#endif
  return formula;
}

std::vector<std::string> MultivariatePolynomialHelper<0, 0, false>::getTerms() const
{
  std::vector<std::string> terms{"par[0]"};
  int indexPar = 1;
  for (unsigned int deg = 1; deg <= mDegree; ++deg) {
    const auto strTmp = combination_with_repetiton<std::vector<std::string>>(deg, mDim, nullptr, indexPar, nullptr, mInteractionOnly);
    terms.insert(terms.end(), strTmp.begin(), strTmp.end());
  }
  return terms;
}

TLinearFitter MultivariatePolynomialHelper<0, 0, false>::getTLinearFitter() const
{
  const std::string formula = getTLinearFitterFormula();
  TLinearFitter fitter(int(mDim), formula.data(), "");
  return fitter;
}

std::vector<float> MultivariatePolynomialHelper<0, 0, false>::fit(TLinearFitter& fitter, std::vector<double>& x, std::vector<double>& y, std::vector<double>& error, const bool clearPoints)
{
  if (clearPoints) {
    fitter.ClearPoints();
  }
  const int nDim = static_cast<int>(x.size() / y.size());
  fitter.AssignData(static_cast<int>(y.size()), nDim, x.data(), y.data(), error.empty() ? nullptr : error.data());

  const int status = fitter.Eval();
  if (status != 0) {
#ifndef GPUCA_NO_FMT
    LOGP(info, "Fitting failed with status: {}", status);
#endif
    return std::vector<float>();
  }

  TVectorD params;
  fitter.GetParameters(params);
  std::vector<float> paramsFloat;
  paramsFloat.reserve(static_cast<unsigned int>(params.GetNrows()));
  std::copy(params.GetMatrixArray(), params.GetMatrixArray() + params.GetNrows(), std::back_inserter(paramsFloat));
  return paramsFloat;
}

std::vector<float> MultivariatePolynomialHelper<0, 0, false>::fit(std::vector<double>& x, std::vector<double>& y, std::vector<double>& error, const bool clearPoints) const
{
  TLinearFitter fitter = getTLinearFitter();
  return fit(fitter, x, y, error, clearPoints);
}

template <class Type>
Type MultivariatePolynomialHelper<0, 0, false>::combination_with_repetiton(const unsigned int degree, const unsigned int dim, const float par[], int& indexPar, const float x[], const bool interactionOnly) const
{
  {
    // each digit represents the currently set dimension
    unsigned int pos[FMaxdegree + 1]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // return value is either the sum of all polynomials or a vector of strings containing the formula for each polynomial
    Type val(0);
    for (;;) {
      // starting on the rightmost digit
      for (unsigned int i = degree; i > 0; --i) {
        // check if digit of current position is at is max position
        if (pos[i] == dim) {
          // increase digit of left position
          ++pos[i - 1];
          // resetting the indices of the digits to the right
          for (unsigned int j = i; j <= degree; ++j) {
            pos[j] = pos[i - 1];
          }
        }
      }

      // check if all combinations are processed
      if (pos[0] == 1) {
        break;
      } else {

        if (interactionOnly) {
          bool checkInteraction = false;
          for (size_t i = 1; i < degree; ++i) {
            checkInteraction = pos[i] == pos[i + 1];
            if (checkInteraction) {
              break;
            }
          }
          if (checkInteraction) {
            ++pos[degree];
            continue;
          }
        }

        if constexpr (std::is_same_v<Type, float>) {
          float term = par[indexPar++];
          for (size_t i = 1; i <= degree; ++i) {
            term *= x[pos[i]];
          }
          val += term;
        } else {
#if !defined(GPUCA_ALIROOT_LIB)
          std::string term{};
          for (size_t i = 1; i <= degree; ++i) {
            term += fmt::format("x[{}] * ", pos[i]);
          }
          term += fmt::format("par[{}]", indexPar++);
          val.emplace_back(term);
#endif
        }
      }

      // increase the rightmost digit
      ++pos[degree];
    }
    return val;
  }
}

float MultivariatePolynomialHelper<0, 0, false>::evalPol(const float par[], const float x[], const unsigned int degree, const unsigned int dim, const bool interactionOnly) const
{
  float val = par[0];
  int indexPar = 1;
  for (unsigned int deg = 1; deg <= degree; ++deg) {
    val += combination_with_repetiton<float>(deg, dim, par, indexPar, x, interactionOnly);
  }
  return val;
}

#endif
