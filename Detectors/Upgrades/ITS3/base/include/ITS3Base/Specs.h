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
/// \brief TDR specs of ITS3
/// \author felix.schlepper@cern.ch

#include "Rtypes.h"

namespace o2::its3
{
namespace constants
{
constexpr double cm{1e+2}; // This is the default unit of TGeo so we use this as scale
constexpr double mu{1e-6 * cm};
constexpr double mm{1e-3 * cm};
namespace pixelarray
{
constexpr double length{9.197 * mm};
constexpr double width{3.571 * mm};
constexpr EColor color{kGreen};
constexpr unsigned int nCols{156};
constexpr unsigned int nRows{440};
constexpr unsigned int nPixels{nCols * nRows};
namespace pixel
{
constexpr double pitchCol{width / static_cast<double>(nCols)};
constexpr double pitchRow{length / static_cast<double>(nRows)};
} // namespace pixel
} // namespace pixelarray
namespace tile
{
namespace biasing
{
constexpr double length{0.06 * mm};
constexpr double width{3.571 * mm};
constexpr EColor color{kYellow};
static_assert(width == pixelarray::width);
} // namespace biasing
namespace powerswitches
{
constexpr double length{9.257 * mm};
constexpr double width{0.02 * mm};
constexpr double z{pixelarray::width};
constexpr EColor color{kBlue};
} // namespace powerswitches
namespace readout
{
constexpr double length{0.525 * mm};
constexpr double width{3.591 * mm};
constexpr EColor color{kMagenta};
static_assert(width == (biasing::width + powerswitches::width));
} // namespace readout
constexpr double width{readout::width};
constexpr double length{powerswitches::length + readout::length};
} // namespace tile
namespace rsu
{
namespace databackbone
{
constexpr double length{9.782 * mm};
constexpr double width{0.06 * mm};
constexpr EColor color{kRed};
} // namespace databackbone
constexpr double length{19.564 * mm};
constexpr double width{21.666 * mm};
} // namespace rsu
namespace segment
{
constexpr double length{rsu::length};
namespace lec
{
constexpr double length{segment::length};
constexpr double width{4.5 * mm};
constexpr EColor color{kCyan};
} // namespace lec
namespace rec
{
constexpr double length{segment::length};
constexpr double width{1.5 * mm};
constexpr EColor color{kCyan};
} // namespace rec
constexpr unsigned int nRSUs{12};
constexpr double width{nRSUs * rsu::width + lec::width + rec::width};
} // namespace segment
constexpr unsigned int nLayers{3};
constexpr std::array<double, nLayers> radii{19 * mm, 25.2 * mm, 31.5 * mm}; // middle radius e.g. inner radius+thickness/2.
constexpr double equatorialGap{1 * mm};
constexpr std::array<unsigned int, nLayers> nSegments{3, 4, 5};
constexpr double thickness{50 * mu};
constexpr double effThickness{66 * mu};
} // namespace constants
} // namespace o2::its3
