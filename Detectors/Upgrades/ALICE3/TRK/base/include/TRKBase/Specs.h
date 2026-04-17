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
/// \brief specs of the ALICE3 TRK

#ifndef O2_ALICE_TRK_SPECS
#define O2_ALICE_TRK_SPECS

#include <array>

#include <math.h>

// This is a temporary version with the specs for the ALICE3 TRK
// This files defines the design specifications of the chips for VD, ML, OT.
// Each TGeoShape has the following properties
// length: dimension in z-axis
// width: dimension in xy-axes
// color: for visualisation
namespace o2::trk::constants
{
// Default unit of TGeo = cm
constexpr double cm{1};
constexpr double mu{1e-4};
constexpr double mm{1e-1};

namespace VD // TODO: add a primitive segmentation with more granularity wrt 1/4 layer = 1 chip
{
namespace silicon
{
constexpr double thickness{20 * mu}; // thickness of the silicon (should be 10 um epitaxial layer + 20 um substrate)?
} // namespace silicon
namespace metalstack
{
constexpr double thickness{80 * mu}; // thickness of the copper metal stack - for the moment it is not implemented. PL: set to 80 um considering silicon as material
} // namespace metalstack
namespace petal
{
constexpr int nLayers{3}; // number of layers in each VD petal
constexpr int nDisks{6};  // number of disks in each VD petal
namespace layer
{
constexpr double pitchX{10 * mu};                                                                                                                        // pitch of the row
constexpr double pitchZ{10 * mu};                                                                                                                        // pitch of the column
constexpr double totalThickness{silicon::thickness + metalstack::thickness};                                                                             // total thickness of the chip
constexpr std::array<double, nLayers> gaps{1.63 * mm, 1.2 * mm, 1.2 * mm};                                                                               // gaps between two consecutive petals
constexpr std::array<double, nLayers> radii{0.5 * cm, 1.2 * cm, 2.5 * cm};                                                                               // radius of layer in cm
constexpr std::array<double, nLayers> width{radii[0] * 2 * M_PI / 4 - gaps[0], radii[1] * 2 * M_PI / 4 - gaps[1], radii[2] * 2 * M_PI / 4 - gaps[2]};    // width of the quarter of layer in cm
constexpr double length{50 * cm};                                                                                                                        // length of the layer
constexpr int nCols{static_cast<int>(length / pitchZ)};                                                                                                  // number of columns in the chip
constexpr std::array<int, nLayers> nRows{static_cast<int>(width[0] / pitchX), static_cast<int>(width[1] / pitchX), static_cast<int>(width[2] / pitchX)}; // number of rows in the chip. For the moment is different for each layer since a siner segmentation in repetitive units is stil to be implemented

} // namespace layer
namespace disk
{ //// TODO: to be filled
constexpr double radiusIn{0.5 * cm};
constexpr double radiusOut{2.5 * cm};
} // namespace disk
} // namespace petal
} // namespace VD

namespace moduleMLOT /// same for ML and OT for the moment
{                    /// TODO: account for different modules in case of changes
namespace silicon
{
constexpr double thickness{100 * mu}; // thickness of the silicon (should be 10 um epitaxial layer + 90 um substrate)?
} // namespace silicon
namespace metalstack
{
constexpr double thickness{0 * mu}; // thickness of the copper metal stack - for the moment it is not implemented
} // namespace metalstack
namespace chip
{
constexpr double width{25 * mm};                                              // width of the chip
constexpr double length{29 * mm};                                             // length of the chip
constexpr double pitchX{20 * mu};                                             // pitch of the row
constexpr double pitchZ{20 * mu};                                             // pitch of the column
constexpr double totalThickness{silicon::thickness + metalstack::thickness};  // total thickness of the chip
static constexpr double passiveEdgeReadOut{1.5 * mm};                         // width of the readout edge -> dead zone
constexpr int nRows{static_cast<int>((width - passiveEdgeReadOut) / pitchX)}; // number of rows in the chip
constexpr int nCols{static_cast<int>(length / pitchZ)};                       // number of columns in the chip
} // namespace chip
namespace gaps
{
constexpr double interChips{50 * mu};          // gap between the chips
constexpr double outerEdgeLongSide{1 * mm};    // gap between the chips and the outer edges (long side)
constexpr double outerEdgeShortSide{0.1 * mm}; // gap between the chips and the outer edges (short side)
} // namespace gaps
constexpr double width{chip::width * 2 + gaps::interChips + 2 * gaps::outerEdgeLongSide};        // width of the module
constexpr double length{chip::length * 4 + 3 * gaps::interChips + 2 * gaps::outerEdgeShortSide}; // length of the module
constexpr int nRows{static_cast<int>(width / chip::pitchX)};                                     // number of columns in the module
constexpr int nCols{static_cast<int>(length / chip::pitchZ)};                                    // number of rows in the module
} // namespace moduleMLOT

namespace ML
{
constexpr int nLayers{5};                                 // number of layers in the ML
constexpr double width{constants::moduleMLOT::width * 1}; // width of the stave
// constexpr double length{constants::moduleMLOT::length * 10};                         // length of the stave
constexpr double length{124 * cm};                                                   // length of the stave, hardcoded to fit the implemented geometry
constexpr int nRows{static_cast<int>(width / constants::moduleMLOT::chip::pitchX)};  // number of rows in the stave
constexpr int nCols{static_cast<int>(length / constants::moduleMLOT::chip::pitchZ)}; // number of columns in the stave
} // namespace ML

namespace OT
{
namespace halfstave
{
constexpr double width{moduleMLOT::width * 1}; // width of the half stave
// constexpr double length{moduleMLOT::length * 20};                         // length of the halfstave
constexpr double length{258 * cm};                                        // length of the halfstave, hardcoded to fit the implemented geometry
constexpr int nRows{static_cast<int>(width / moduleMLOT::chip::pitchX)};  // number of rows in the halfstave
constexpr int nCols{static_cast<int>(length / moduleMLOT::chip::pitchZ)}; // number of columns in the halfstave
} // namespace halfstave
constexpr int nLayers{3};                                                 // number of layers in the OT
constexpr double width{halfstave::width * 2};                             // width of the stave
constexpr double length{halfstave::length};                               // length of the stave
constexpr int nRows{static_cast<int>(width / moduleMLOT::chip::pitchX)};  // number of rows in the stave
constexpr int nCols{static_cast<int>(length / moduleMLOT::chip::pitchZ)}; // number of columns in the stave
} // namespace OT

namespace apts /// parameters for the APTS response
{
constexpr double pitchX{15.0 * mu};
constexpr double pitchZ{15.0 * mu};
constexpr double responseYShift{15.5 * mu};
constexpr double thickness{45 * mu};
} // namespace apts

namespace alice3resp /// parameters for the alice3 chip response
{
constexpr double pitchX{10.0 * mu};
constexpr double pitchZ{10.0 * mu};
constexpr double responseYShift{5 * mu}; /// center of the epitaxial layer
constexpr double thickness{20 * mu};
} // namespace alice3resp
} // namespace o2::trk::constants

#endif
