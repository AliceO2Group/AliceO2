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

/// \file   Constants.h
/// \brief  General constants in FV0
///
/// \author Maciej Slupecki, University of Jyvaskyla, Finland

#ifndef ALICEO2_FD3_CONSTANTS_
#define ALICEO2_FD3_CONSTANTS_

namespace o2
{
namespace fd3
{
struct Constants {
  static constexpr unsigned int nsect = 8;
  static constexpr unsigned int nringsA = 5;
  static constexpr unsigned int nringsC = 6;

  static constexpr float etaMax = 7.0f;
  static constexpr float etaMin = 4.0f;

  static constexpr unsigned int nringsA_withMG = 3;
  static constexpr float etaMinA_withMG = 5.0f;
};

} // namespace fd3
} // namespace o2
#endif
