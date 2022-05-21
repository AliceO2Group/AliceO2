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

using namespace GPUCA_NAMESPACE::gpu;

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
void MultivariatePolynomialHelper<0, 0>::print() const
{
#ifndef GPUCA_NO_FMT
  const auto terms = getTerms();
  std::string formula = "";
  for (int i = 0; i < (int)terms.size() - 1; ++i) {
    formula += fmt::format("{} + ", terms[i]);
  }
  formula += terms.back();
  LOGP(info, formula.c_str());
#endif
}

std::vector<std::string> MultivariatePolynomialHelper<0, 0>::getTerms() const
{
  std::vector<std::string> terms{"par[0]"};
  int indexPar = 1;
  for (unsigned int deg = 1; deg <= mDegree; ++deg) {
    const auto strTmp = combination_with_repetiton<std::vector<std::string>>(deg, mDim, nullptr, indexPar, nullptr);
    terms.insert(terms.end(), strTmp.begin(), strTmp.end());
  }
  return terms;
}
#endif
