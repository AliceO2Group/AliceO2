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

/// \file ClustererParam.h
/// \brief Definition of the ITS/MFT clusterer settings

#ifndef ALICEO2_ITSMFTCLUSTERERPARAM_H_
#define ALICEO2_ITSMFTCLUSTERERPARAM_H_

#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string_view>
#include <string>

namespace o2
{
namespace itsmft
{
template <int N>
struct ClustererParam : public o2::conf::ConfigurableParamHelper<ClustererParam<N>> {
  static_assert(N == o2::detectors::DetID::ITS || N == o2::detectors::DetID::MFT, "only DetID::ITS or DetID:: MFT are allowed");

  static constexpr std::string_view getParamName()
  {
    return N == o2::detectors::DetID::ITS ? ParamName[0] : ParamName[1];
  }

  int maxRowColDiffToMask = DEFRowColDiffToMask(); ///< pixel may be masked as overflow if such a neighbour in prev frame was fired
  int maxBCDiffToMaskBias = 10;                    ///< mask if 2 ROFs differ by <= StrobeLength + Bias BCs, use value <0 to disable masking

  O2ParamDef(ClustererParam, getParamName().data());

 private:
  static constexpr int DEFRowColDiffToMask()
  {
    // default neighbourhood definition
    return N == o2::detectors::DetID::ITS ? 1 : 1; // ITS and MFT will suppress also closest neigbours
  }
  static constexpr std::string_view ParamName[2] = {"ITSClustererParam", "MFTClustererParam"};
};

template <int N>
ClustererParam<N> ClustererParam<N>::sInstance;

} // namespace itsmft

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>> : std::true_type {
};
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::itsmft::ClustererParam<o2::detectors::DetID::MFT>> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif
