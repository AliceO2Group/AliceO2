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

/// \file   BoostHistogramUtils.cxx
/// \author Hannah Bossi, hannah.bossi@yale.edu

#include "CommonUtils/BoostHistogramUtils.h"

namespace o2
{
namespace utils
{

std::string createErrorMessageFitGaus(o2::utils::FitGausError_t errorcode)
{
  switch (errorcode) {
    case FitGausError_t::FIT_ERROR_MIN:
      return "Gaus fit failed! xMax < 4";
    case FitGausError_t::FIT_ERROR_MAX:
      return "Gaus fit failed! xMax too large";
    case FitGausError_t::FIT_ERROR_ENTRIES:
      return "Gaus fit failed! entries < 12";
    case FitGausError_t::FIT_ERROR_KTOL_MEAN:
      return "Gaus fit failed! std::abs(par[1]) < kTol";
    case FitGausError_t::FIT_ERROR_KTOL_SIGMA:
      return "Gaus fit failed! std::abs(par[2]) < kTol";
    case FitGausError_t::FIT_ERROR_KTOL_RMS:
      return "Gaus fit failed! RMS < kTol";
    default:
      return "Gaus fit failed! Unknown error code";
  }
  return "Gaus fit failed! Unknown error code";
}

/// \brief Printing an error message when then fit returns an invalid result
/// \param errorcode Error of the type FitGausError_t, thrown when fit result is invalid.
std::string createErrorMessage(o2::utils::FitGausError_t errorcode)
{
  return "[Error]: Fit return an invalid result.";
}
} // namespace utils
} // namespace o2