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

// This files defines the design specifications of the chip.
// Each TGeoShape has the following properties
// length: dimension in z-axis
// width: dimension in xy-axes
// color: for visulisation
namespace o2::its3::constants
{
constexpr double cm{1.0}; // This is the default unit of TGeo so we use this as scale
constexpr double mu{1e-4 * cm};
constexpr double mm{1e-1 * cm};
namespace pixelarray
{
constexpr double width{9.197 * mm};
constexpr double length{3.571 * mm};
constexpr int nCols{156};
constexpr int nRows{442};
constexpr int nPixels{nRows * nCols};
constexpr EColor color{kGreen};
constexpr double area{width * length};
} // namespace pixelarray
namespace tile
{
namespace biasing
{
constexpr double width{0.06 * mm};
constexpr double length{3.571 * mm};
constexpr EColor color{kYellow};
static_assert(length == pixelarray::length);
} // namespace biasing
namespace powerswitches
{
constexpr double width{9.257 * mm};
constexpr double length{0.02 * mm};
constexpr double z{pixelarray::width};
constexpr EColor color{kBlue};
} // namespace powerswitches
namespace readout
{
constexpr double width{0.525 * mm};
constexpr double length{3.591 * mm};
constexpr EColor color{kMagenta};
static_assert(length == (biasing::length + powerswitches::length));
} // namespace readout
constexpr double length{readout::length};
constexpr double width{powerswitches::width + readout::width};
} // namespace tile
namespace rsu
{
namespace databackbone
{
constexpr double width{9.782 * mm};
constexpr double length{0.06 * mm};
constexpr EColor color{kRed};
} // namespace databackbone
constexpr double width{19.564 * mm};
constexpr double length{21.666 * mm};
constexpr unsigned int nTiles{12};
} // namespace rsu
namespace segment
{
constexpr double width{rsu::width};
namespace lec
{
constexpr double width{segment::width};
constexpr double length{4.5 * mm};
constexpr EColor color{kCyan};
} // namespace lec
namespace rec
{
constexpr double width{segment::width};
constexpr double length{1.5 * mm};
constexpr EColor color{kCyan};
} // namespace rec
constexpr unsigned int nRSUs{12};
constexpr unsigned int nTilesPerSegment{nRSUs * rsu::nTiles};
constexpr double length{(nRSUs * rsu::length) + lec::length + rec::length};
constexpr double lengthSensitive{nRSUs * rsu::length};
} // namespace segment
namespace carbonfoam
{
// TODO: Waiting for the further information from WP5(Corrado)
constexpr double HringLength{6.0 * mm};                                    // from blueprint
constexpr double longeronsWidth{2.0 * mm};                                 // what is the height of the longerons?
constexpr double longeronsLength{segment::length - (2 * HringLength)};     // 263mm from blueprint; overrriden to be consitent
constexpr double edgeBetwChipAndFoam{1.0 * mm};                            // from blueprint but not used cause forms are already overlapping
constexpr double gapBetwHringsLongerons{0.05 * mm};                        // from blueprint
constexpr std::array<int, 3> nHoles{11, 11, 11};                           // how many holes for each layer?
constexpr std::array<double, 3> radiusHoles{1.0 * mm, 1.0 * mm, 2.0 * mm}; // TODO what is the radius of the holes for each layer?
constexpr double thicknessOuterFoam{7 * mm};                               // TODO: lack of carbon foam radius for layer 2, use 0.7 cm as a temporary value
constexpr EColor color{kGray};
} // namespace carbonfoam
namespace metalstack
{
constexpr double thickness{5 * mu}; // physical thickness of the copper metal stack
constexpr double length{segment::length};
constexpr double width{segment::width};
constexpr EColor color{kBlack};
} // namespace metalstack
namespace silicon
{
constexpr double thickness{45 * mu};                                     // thickness of silicon
constexpr double thicknessIn{(thickness + metalstack::thickness) / 2.};  // inner silicon thickness
constexpr double thicknessOut{(thickness - metalstack::thickness) / 2.}; // outer silicon thickness
} // namespace silicon
constexpr unsigned int nLayers{3};
constexpr unsigned int nTotLayers{7};
constexpr unsigned int nSensorsIB{2 * nLayers};
constexpr double equatorialGap{1 * mm};
constexpr std::array<unsigned int, nLayers> nSegments{3, 4, 5};
constexpr double totalThickness{silicon::thickness + metalstack::thickness};                                                                                         // total chip thickness
constexpr std::array<double, nLayers> radii{19.0006 * mm, 25.228 * mm, 31.4554 * mm};                                                                                // nominal radius
constexpr std::array<double, nLayers> radiiInner{radii[0] - silicon::thicknessIn, radii[1] - silicon::thicknessIn, radii[2] - silicon::thicknessIn};                 // inner silicon radius
constexpr std::array<double, nLayers> radiiOuter{radii[0] + silicon::thicknessOut, radii[1] + silicon::thicknessOut, radii[2] + silicon::thicknessOut};              // outer silicon radius
constexpr std::array<double, nLayers> radiiMiddle{(radiiInner[0] + radiiOuter[0]) / 2., (radiiInner[1] + radiiOuter[1]) / 2., (radiiInner[2] + radiiOuter[2]) / 2.}; // middle silicon radius

// extra information of pixels and their response functions
namespace pixelarray::pixels
{
namespace mosaix
{
constexpr double pitchX{width / static_cast<double>(nRows)};
constexpr double pitchZ{length / static_cast<double>(nCols)};
} // namespace mosaix
namespace apts
{
constexpr double pitchX{15.0 * mu};
constexpr double pitchZ{15.0 * mu};
constexpr double responseYShift{15.5 * mu};
} // namespace apts
namespace moss
{
namespace top
{
constexpr double pitchX{22.5 * mu};
constexpr double pitchZ{22.5 * mu};
} // namespace top
namespace bot
{
constexpr double pitchX{18.0 * mu};
constexpr double pitchZ{18.0 * mu};
} // namespace bot
} // namespace moss
} // namespace pixelarray::pixels

namespace detID
{
constexpr unsigned int mDetIDs{2 * 12 * 12 * 12};                  //< 2 Hemispheres * (3,4,5=12 segments in a layer) * 12 RSUs in a segment * 12 Tiles in a RSU
constexpr unsigned int l0IDStart{0};                               //< Start DetID layer 0
constexpr unsigned int l0IDEnd{(2 * 3 * 12 * 12) - 1};             //< End First DetID layer 0; inclusive range
constexpr unsigned int l0IDTot{2 * 3 * 12 * 12};                   //< Total DetID in Layer 0
constexpr unsigned int l1IDStart{l0IDEnd + 1};                     //< Start DetID layer 1
constexpr unsigned int l1IDEnd{l1IDStart + (2 * 4 * 12 * 12) - 1}; //< End First DetID layer 1; inclusive range
constexpr unsigned int l1IDTot{2 * 4 * 12 * 12};                   //< Total DetID in Layer 1
constexpr unsigned int l2IDStart{l1IDEnd + 1};                     //< Start DetID layer 2
constexpr unsigned int l2IDEnd{l2IDStart + (2 * 5 * 12 * 12) - 1}; //< End First DetID layer 2; inclusive range
constexpr unsigned int l2IDTot{2 * 5 * 12 * 12};                   //< Total DetID in Layer 2
constexpr unsigned int nChips{l2IDEnd + 1};                        //< number of Chips (PixelArrays) in IB

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
  return detID < static_cast<T>(nChips) && detID >= 0;
}

} // namespace detID

// services
namespace services
{
// FIXME these value are hallucinated since this not yet defined
constexpr double thickness{2.2 * mm};                                         // thickness of structure
constexpr double radiusInner{radiiOuter[2] + carbonfoam::thicknessOuterFoam}; // inner radius of services
constexpr double radiusOuter{radiusInner + thickness};                        // outer radius of services
constexpr double length{segment::length + (1 * cm)};                          // length
constexpr EColor color{kBlue};
} // namespace services

} // namespace o2::its3::constants

#endif
