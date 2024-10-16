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

/// \file Specs.h
/// \brief TDR V2 specs of ITS3
/// \author felix.schlepper@cern.ch
/// \author chunzheng.wang@cern.ch

#ifndef O2_ALICE_ITS3_SPECS
#define O2_ALICE_ITS3_SPECS

#include "Rtypes.h"

#include <array>

namespace o2::its3::constants
{
constexpr float cm{1e+2}; // This is the default unit of TGeo so we use this as scale
constexpr float mu{1e-6 * cm};
constexpr float mm{1e-3 * cm};
namespace pixelarray
{
constexpr float width{9.197 * mm};
constexpr float length{3.571 * mm};
constexpr int nCols{156};
constexpr int nRows{442};
constexpr int nPixels{nRows * nCols};
constexpr EColor color{kGreen};
constexpr float area{width * length};
} // namespace pixelarray
namespace tile
{
namespace biasing
{
constexpr float width{0.06 * mm};
constexpr float length{3.571 * mm};
constexpr EColor color{kYellow};
static_assert(length == pixelarray::length);
} // namespace biasing
namespace powerswitches
{
constexpr float width{9.257 * mm};
constexpr float length{0.02 * mm};
constexpr float z{pixelarray::width};
constexpr EColor color{kBlue};
} // namespace powerswitches
namespace readout
{
constexpr float width{0.525 * mm};
constexpr float length{3.591 * mm};
constexpr EColor color{kMagenta};
static_assert(length == (biasing::length + powerswitches::length));
} // namespace readout
constexpr float length{readout::length};
constexpr float width{powerswitches::width + readout::width};
} // namespace tile
namespace rsu
{
namespace databackbone
{
constexpr float width{9.782 * mm};
constexpr float length{0.06 * mm};
constexpr EColor color{kRed};
} // namespace databackbone
constexpr float width{19.564 * mm};
constexpr float length{21.666 * mm};
constexpr unsigned int nTiles{12};
} // namespace rsu
namespace segment
{
constexpr float width{rsu::width};
namespace lec
{
constexpr float width{segment::width};
constexpr float length{4.5 * mm};
constexpr EColor color{kCyan};
} // namespace lec
namespace rec
{
constexpr float width{segment::width};
constexpr float length{1.5 * mm};
constexpr EColor color{kCyan};
} // namespace rec
constexpr unsigned int nRSUs{12};
constexpr unsigned int nTilesPerSegment{nRSUs * rsu::nTiles};
constexpr float length{nRSUs * rsu::length + lec::length + rec::length};
constexpr float lengthSensitive{nRSUs * rsu::length};
} // namespace segment
namespace carbonfoam
{
// TODO: Waiting for the further information from WP5(Corrado)
constexpr float longeronsWidth{2.0 * mm};                                 // what is the height of the longerons?
constexpr float longeronsLength{263 * mm};                                // from blueprint
constexpr float HringLength{6.0 * mm};                                    // from blueprint
constexpr float edgeBetwChipAndFoam{1.0 * mm};                            // from blueprint but not used cause forms are already overlapping
constexpr float gapBetwHringsLongerons{0.05 * mm};                        // from blueprint
constexpr std::array<int, 3> nHoles{11, 11, 11};                          // how many holes for each layer?
constexpr std::array<float, 3> radiusHoles{1.0 * mm, 1.0 * mm, 2.0 * mm}; // what is the radius of the holes for each layer?
constexpr EColor color{kGray};
} // namespace carbonfoam
constexpr unsigned int nLayers{3};
constexpr unsigned int nTotLayers{7};
constexpr unsigned int nSensorsIB{2 * nLayers};
constexpr float equatorialGap{1 * mm};
constexpr std::array<unsigned int, nLayers> nSegments{3, 4, 5};
constexpr float thickness{50 * mu};                                                                                                  //< Physical Thickness of chip
constexpr float effThickness{66 * mu};                                                                                               //< Physical thickness + metal substrate
constexpr std::array<float, nLayers> radii{19.0006 * mm, 25.228 * mm, 31.4554 * mm};                                                 // middle radius e.g. inner radius+thickness/2.
constexpr std::array<float, nLayers> radiiInner{radii[0] - thickness / 2.f, radii[1] - thickness / 2.f, radii[2] - thickness / 2.f}; // inner radius
constexpr std::array<float, nLayers> radiiOuter{radii[0] + thickness / 2.f, radii[1] + thickness / 2.f, radii[2] + thickness / 2.f}; // inner radius
namespace detID
{
constexpr unsigned int mDetIDs{2 * 12 * 12 * 12};                //< 2 Hemispheres * (3,4,5=12 segments in a layer) * 12 RSUs in a segment * 12 Tiles in a RSU
constexpr unsigned int l0IDStart{0};                             //< Start DetID layer 0
constexpr unsigned int l0IDEnd{2 * 3 * 12 * 12 - 1};             //< End First DetID layer 0; inclusive range
constexpr unsigned int l0IDTot{2 * 3 * 12 * 12};                 //< Total DetID in Layer 0
constexpr unsigned int l1IDStart{l0IDEnd + 1};                   //< Start DetID layer 1
constexpr unsigned int l1IDEnd{l1IDStart + 2 * 4 * 12 * 12 - 1}; //< End First DetID layer 1; inclusive range
constexpr unsigned int l1IDTot{2 * 4 * 12 * 12};                 //< Total DetID in Layer 1
constexpr unsigned int l2IDStart{l1IDEnd + 1};                   //< Start DetID layer 2
constexpr unsigned int l2IDEnd{l2IDStart + 2 * 5 * 12 * 12 - 1}; //< End First DetID layer 2; inclusive range
constexpr unsigned int l2IDTot{2 * 5 * 12 * 12};                 //< Total DetID in Layer 2
constexpr unsigned int nChips{l2IDEnd + 1};                      //< number of Chips (PixelArrays) in IB

template <typename T = int>
inline T getDetID2Layer(T detID)
{
  if (static_cast<T>(l0IDStart) <= detID && detID <= static_cast<T>(l0IDEnd)) {
    return 0;
  } else if (static_cast<T>(l1IDStart) <= detID && detID <= static_cast<T>(l1IDEnd)) {
    return 1;
  } else if (static_cast<T>(l2IDStart) <= detID && detID <= static_cast<T>(l2IDEnd)) {
    return 2;
  }
  return -1;
}

template <typename T = int>
inline T getSensorID(T detID)
{
  auto layer = getDetID2Layer(detID);
  if (layer == 0) {
    return ((detID - l0IDStart) < static_cast<T>(l0IDTot) / 2) ? 0 : 1;
  } else if (layer == 1) {
    return ((detID - l1IDStart) < static_cast<T>(l1IDTot) / 2) ? 2 : 3;
  } else if (layer == 2) {
    return ((detID - l2IDStart) < static_cast<T>(l2IDTot) / 2) ? 4 : 5;
  }
  return -1;
}

template <typename T = int>
inline bool isDetITS3(T detID)
{
  return detID < static_cast<T>(nChips);
}

} // namespace detID
} // namespace o2::its3::constants

#endif
