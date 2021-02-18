// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClustererParam.h
/// \brief Definition of the ITS/MFT clusterer settings

#ifndef ALICEO2_ITSMFTCLUSTERERPARAM_H_
#define ALICEO2_ITSMFTCLUSTERERPARAM_H_

#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string_view>

namespace o2
{
namespace itsmft
{
template <int N>
struct ClustererParam : public o2::conf::ConfigurableParamHelper<ClustererParam<N>> {
  static_assert(N == o2::detectors::DetID::ITS || N == o2::detectors::DetID::MFT, "only DetID::ITS orDetID:: MFT are allowed");

  static constexpr std::string_view getParamName()
  {
    return N == o2::detectors::DetID::ITS ? ParamName[0] : ParamName[1];
  }

  int maxRowColDiffToMask = 0;  ///< pixel may be masked as overflow if such a neighbour in prev frame was fired
  int maxBCDiffToMaskBias = 10; ///< mask if 2 ROFs differ by <= StrobeLength + Bias BCs, use value <0 to disable masking

  O2ParamDef(ClustererParam, getParamName().data());

 private:
  static constexpr std::string_view ParamName[2] = {"ITSClustererParam", "MFTClustererParam"};
};

template <int N>
ClustererParam<N> ClustererParam<N>::sInstance;

} // namespace itsmft
} // namespace o2

#endif
