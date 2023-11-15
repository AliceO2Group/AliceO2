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
///
/// \file Constants.h
/// \brief Some constants, fixed parameters and look-up-table functions
///

#ifndef O2_MFT_CONSTANTS_H_
#define O2_MFT_CONSTANTS_H_

#include <climits>
#include <vector>
#include <array>

#include <Rtypes.h>

#include "CommonConstants/MathConstants.h"

namespace o2
{
namespace mft
{

namespace constants
{

namespace mft
{
constexpr Int_t LayersNumber{10};
constexpr Int_t DisksNumber{5};
constexpr Int_t TrackletsPerRoad{LayersNumber - 1};
constexpr Int_t CellsPerRoad{LayersNumber - 2};
constexpr Int_t ClustersPerCell{2};
constexpr Int_t UnusedIndex{-1};
constexpr Float_t Resolution{0.0005f};
constexpr std::array<Float_t, LayersNumber> LayerZCoordinate()
{
  return std::array<Float_t, LayersNumber>{-45.3, -46.7, -48.6, -50.0, -52.4, -53.8, -67.7, -69.1, -76.1, -77.5};
}
constexpr std::array<Float_t, LayersNumber> InverseLayerZCoordinate()
{
  return std::array<Float_t, LayersNumber>{-1. / 45.3, -1. / 46.7, -1. / 48.6, -1. / 50.0, -1. / 52.4, -1. / 53.8, -1. / 67.7, -1. / 69.1, -1. / 76.1, -1. / 77.5};
}
constexpr Int_t MaxPointsInRoad{100};
constexpr Int_t MaxCellsInRoad{100};
} // namespace mft

namespace index_table
{
constexpr std::array<Float_t, o2::mft::constants::mft::LayersNumber> RMin{2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 3.1, 3.1, 3.5, 3.5}; // [cm]
constexpr std::array<Float_t, o2::mft::constants::mft::LayersNumber> RMax{12.5, 12.5, 12.5, 12.5, 14.0, 14.0, 17.0, 17.0, 17.5, 17.5};

constexpr Float_t PhiMin{0.};
constexpr Float_t PhiMax{o2::constants::math::TwoPI}; // [rad]

constexpr Int_t MaxRPhiBins{120 * 30 + 1};
} // namespace index_table

} // namespace constants
} // namespace mft
} // namespace o2

#endif /* O2_MFT_CONSTANTS_H_ */
