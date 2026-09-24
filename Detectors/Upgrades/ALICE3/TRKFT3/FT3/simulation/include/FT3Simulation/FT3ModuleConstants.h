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

/// \file FT3ModuleConstants.h
/// \brief Definition of various constants for tiling the modules of sensors
///
/// The materials themselves live in FT3Materials.h.

#ifndef FT3MODULECONSTANTS_H
#define FT3MODULECONSTANTS_H

#include <vector>
#include <map>
#include <TMath.h>

namespace o2::ft3::ModuleConstants
{
/* CURRENT STATUS:
 * 25x29mm sensors, 2mm inactive on one side
 * Most granular layout is 2x1 sensors, where the one on the right has the inactive region
 * on the right, and the one on the left has the inactive region on the left.
 * When stacking 2x1 modules, there is a 0.2mm gap between them. By default, we assume this
 * gap to be ABOVE the most recently placed module.
 *
 * |<- 25mm ->||<- 25mm ->|
 * _______________________
 * ------------------------  0.2mm gap above
 * | |        ||        | |
 * | |        ||        | |
 * | |        ||        | |
 * | |        ||        | |  29mm sensor height
 * | |        ||        | |
 * | |        ||        | |
 * ------------------------
 *            ^
 *            |
 *   0.2mm gap in the middle
 */
// First set all layout constants for the rest of the function
const double single_sensor_width = 2.5;
const double single_sensor_height = 2.9;
const double inactive_width = 0.2;
const double sensor2x1_gap = 0.02;     // both between L&R sensors in 2x1, and between two 2x1s
const double stackGap = sensor2x1_gap; // gap between 2xN module stacks

const double active_width = single_sensor_width - inactive_width;
const double active_height = single_sensor_height;

const double sensor2x1_width = 2 * single_sensor_width;
const double sensor2x1_active_width = 2 * active_width;
const double sensor2x1_height = single_sensor_height;
const std::vector<unsigned> kSensorsPerStack = {4, 2, 1};
inline const double getStackHeight(unsigned nSensorsPerStack)
{
  return nSensorsPerStack * sensor2x1_height +
         (nSensorsPerStack - 1) * sensor2x1_gap;
}

// small helper function to get 1-indexed stave ID, counting from the middle outwards,
// with negative IDs on the left and positive IDs on the right
inline const int staveIdxToID(int staveIdx, unsigned nStavesPerDisc)
{
  unsigned nStavesOneSide = nStavesPerDisc / 2;
  bool isRight = staveIdx >= nStavesOneSide;
  return staveIdx - nStavesOneSide + isRight;
}

/*
 * Stave x midpoints follow from their number and spacing: they are spread
 * symmetrically about x=0, so an even count leaves a gap on the axis rather
 * than putting a stave on it. Deriving them keeps the spacing and the
 * positions from drifting apart.
 */
inline std::vector<double> makeStaveXMidpoints(unsigned nStaves, double spacing)
{
  std::vector<double> midpoints(nStaves);
  for (unsigned i = 0; i < nStaves; i++) {
    midpoints[i] = (i - (nStaves - 1) / 2.0) * spacing;
  }
  return midpoints;
}

/*
 * Staves alternate between the front and the back of the disc so that
 * neighbours can overlap in x without touching, staggered in z by
 * z_offsetStave. Starting at 0 puts the leftmost stave at the back.
 */
inline std::vector<bool> makeStaveOnFront(unsigned nStaves)
{
  std::vector<bool> staveOnFront(nStaves);
  for (unsigned i = 0; i < nStaves; i++) {
    staveOnFront[i] = i % 2;
  }
  return staveOnFront;
}

// material properties
const double siliconThickness = 0.01;
const double copperThickness = 0.006;
const double kaptonThickness = 0.03;
const double epoxyThickness = 0.0012;

const double effectiveCarbonThickness_Stave = 0.02; // foam + shell
const double staveOpeningAngle = 60 * TMath::DegToRad();
const double sinTheta = TMath::Sin(staveOpeningAngle / 2);
const double alpha = TMath::Pi() / 2 - staveOpeningAngle / 2; // bottom angles
const double staveSensorGap = 0.1;                            // 2mm padding on each side when sensor is glued
const double staveTriangleHeight = (sensor2x1_width + 2 * staveSensorGap) / 2.0 / tan(staveOpeningAngle / 2.0);
/*
 * Now describe the offset of every other stave in z to avoid overlaps
 * ______      ______
 * \    /______\    / | <-- z_offsetStave
 *  \  / \    / \  /
 *   \/   \  /   \/
 *         \/
 */
// If midpoint spacing becomes non constant, this becomes a function
// TODO: add some tolerance to avoid overlaps?
inline const double z_offsetStave(double x_midpoint_spacing)
{
  return staveTriangleHeight *
         (2 - x_midpoint_spacing / (sensor2x1_width / 2 + staveSensorGap));
}

/*
 * One uninterrupted fill of 2xN modules along a stave.
 *
 * yStart is the y of the BOTTOM edge of the first module; modules follow
 * upwards, each separated from the previous one by stackGap. Nothing is
 * mirrored: the layout is symmetric about the y-axis (stave +-ID) but NOT
 * about the x-axis, so every fill states its own y explicitly.
 *
 * A stave that the beam pipe cuts in two therefore carries two fills, one
 * below the hole and one above it, each with its own yStart.
 */
struct StaveFill {
  const double yStart;
  const std::vector<unsigned> stackHeights;
};

// Struct for stave position configuration (varies between ML/OT)
struct StaveConfig {
  const unsigned isML; // whether this config is for ML or OT
  /*
   * Constants for staves are written for both positive
   * and negative x even though they are just mirrored now,
   * because there might be design changes in the future
   * that require a non-mirrored layout, making it easier to
   * change here if so required, even though it looks uglier now.
   *
   * The second element in the mapping pair is whether the stave
   * with a certain ID should be mirrored around the x-axis.
   */
  // map from Stave ID (1-indexed from other documents) to midpoint
  // Do NOT add any zero midpoints, this is taken off separately
  const std::map<int, std::pair<double, bool>>& staveID_to_y_midpoint;
  // lengths of staves, their midpoint, and their face
  const std::vector<double>& y_lengths;
  const std::vector<double>& x_midpoints;
  const double x_midpoint_spacing;
  // whether staves can be placed outside of nominal radii
  const double maxToleranceInner;
  const double maxToleranceOuter;
  // which side of the disc do we place the stave?
  // kSegmentedStave: staggering staves in z (see z_offsetStave)
  // accessed via stave index, NOT stave ID
  const std::vector<bool>& staveOnFront;
  /*
   * Tabulated module layout, used when FT3Base.useExactStavePlacement is set.
   * One entry per stave, indexed like x_midpoints (NOT by stave ID), holding
   * that stave's fills: one for a stave reaching across y=0, two for a stave
   * split by the beam pipe.
   */
  const std::vector<std::vector<StaveFill>>& exactStaveFills;
};

namespace OT_StavePositions
{
/*
 * Staves that the beam pipe cuts in two, built as two pieces on +-y_midpoint.
 * Do NOT add any zero midpoints, this is taken off separately.
 */
const std::map<int, std::pair<double, bool>> staveID_to_y_midpoint = {
  {-5, {32.73, true}},
  {-4, {38.609, true}},
  {-3, {42.094, true}},
  {-2, {43.023, true}},
  {-1, {44.054, true}},
  {1, {44.053, true}},
  {2, {43.023, true}},
  {3, {42.094, true}},
  {4, {38.609, true}},
  {5, {32.73, true}}};
/*
 * Length of one stave piece: for a stave in staveID_to_y_midpoint that is one
 * of its two pieces, otherwise the whole stave centred on y=0. Sized to cover
 * the modules in exactStaveFills, with 1 mm of margin.
 */
const std::vector<double> y_lengths = {
  40.87, 61.31, 78.83, 90.51, 99.27,
  108.03, 113.87, 119.71, 122.63, 128.47,
  64.23, 55.47, 49.63, 49.63, 47.23,
  47.23, 49.63, 49.63, 55.47, 64.23,
  128.47, 122.63, 119.71, 113.87, 108.03,
  99.27, 90.51, 78.83, 61.31, 40.87};
const unsigned nStaves = 30; // y_lengths, staveOnFront and exactStaveFills follow this
const double x_midpoint_spacing = 4.5;
const std::vector<double> x_midpoints = makeStaveXMidpoints(nStaves, x_midpoint_spacing);
const double maxToleranceInner = 9.;   // close but not directly at 10cm yet
const double maxToleranceOuter = 3.4;  // leave 1mm for layer air encapsulation
const std::vector<bool> staveOnFront = makeStaveOnFront(nStaves);
/*
 * From the disk optimiser: Rin 20, Rout 68, stave width 5.22, overlap 0.72,
 * intrusion <= 9, extrusion <= 3, giving 99.84% filling with staves and
 * 99.29% with modules. One entry per stave in x_midpoints order; a stave cut
 * in two by the beam pipe has one fill either side of the hole.
 *
 * Each fill is anchored on its centre rather than the optimiser's own yStart,
 * because FT3 puts a stackGap between modules and the optimiser does not.
 */
const std::vector<std::vector<StaveFill>> exactStaveFills = {
  {{-20.43, {4, 4, 3, 3}}}, // ID -15
  {{-30.65, {4, 4, 4, 3, 3, 3}}}, // ID -14
  {{-39.41, {4, 4, 4, 4, 4, 4, 3}}}, // ID -13
  {{-45.25, {4, 4, 4, 4, 4, 4, 4, 3}}}, // ID -12
  {{-49.63, {4, 4, 4, 4, 4, 4, 4, 3, 3}}}, // ID -11
  {{-54.01, {4, 4, 4, 4, 4, 4, 4, 3, 3, 3}}}, // ID -10
  {{-56.93, {4, 4, 4, 4, 4, 4, 4, 4, 4, 3}}}, // ID -9
  {{-59.85, {4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3}}}, // ID -8
  {{-61.31, {4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3}}}, // ID -7
  {{-64.23, {4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}}}, // ID -6
  {{-64.84, {4, 4, 4, 4, 3, 3}}, {0.62, {4, 4, 4, 4, 3, 3}}}, // ID -5
  {{-66.339, {4, 4, 4, 4, 3}}, {10.879, {4, 4, 4, 4, 3}}}, // ID -4
  {{-66.904, {4, 4, 3, 3, 3}}, {17.284, {4, 4, 3, 3, 3}}}, // ID -3
  {{-67.833, {4, 4, 3, 3, 3}}, {18.213, {4, 4, 3, 3, 3}}}, // ID -2
  {{-67.147, {4, 3, 3, 3, 3}}, {20.96, {4, 3, 3, 3, 3}}}, // ID -1
  {{-67.66, {4, 3, 3, 3, 3}}, {20.447, {4, 3, 3, 3, 3}}}, // ID +1
  {{-67.833, {4, 4, 3, 3, 3}}, {18.213, {4, 4, 3, 3, 3}}}, // ID +2
  {{-66.904, {4, 4, 3, 3, 3}}, {17.284, {4, 4, 3, 3, 3}}}, // ID +3
  {{-66.339, {4, 4, 4, 4, 3}}, {10.879, {4, 4, 4, 4, 3}}}, // ID +4
  {{-64.84, {4, 4, 4, 4, 3, 3}}, {0.62, {4, 4, 4, 4, 3, 3}}}, // ID +5
  {{-64.23, {4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}}}, // ID +6
  {{-61.31, {4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3}}}, // ID +7
  {{-59.85, {4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3}}}, // ID +8
  {{-56.93, {4, 4, 4, 4, 4, 4, 4, 4, 4, 3}}}, // ID +9
  {{-54.01, {4, 4, 4, 4, 4, 4, 4, 3, 3, 3}}}, // ID +10
  {{-49.63, {4, 4, 4, 4, 4, 4, 4, 3, 3}}}, // ID +11
  {{-45.25, {4, 4, 4, 4, 4, 4, 4, 3}}}, // ID +12
  {{-39.41, {4, 4, 4, 4, 4, 4, 3}}}, // ID +13
  {{-30.65, {4, 4, 4, 3, 3, 3}}}, // ID +14
  {{-20.43, {4, 4, 3, 3}}}, // ID +15
};
} // namespace OT_StavePositions

namespace ML_StavePositions
{
// Use prelim numbers for now, these will change! TODO
const std::map<int, std::pair<double, bool>> staveID_to_y_midpoint = {
  {-3, {19.1, true}},
  {-2, {21.8, true}},
  {-1, {22.5, true}},
  {1, {22.5, true}},
  {2, {21.8, true}},
  {3, {19.1, true}}};
const std::vector<double> y_lengths = {
  30.5, 44.5, 53.6, 60.0, 64.6, 29.5, 25.8, 25.0,
  25.0, 25.8, 29.5, 64.6, 60.0, 53.6, 44.5, 30.5};
const unsigned nStaves = 16; // y_lengths, staveOnFront and exactStaveFills follow this
const double x_midpoint_spacing = 4.5;
const std::vector<double> x_midpoints = makeStaveXMidpoints(nStaves, x_midpoint_spacing);
const double maxToleranceInner = 0.;  // default not allowed inwards
const double maxToleranceOuter = 3.4; // leave 1mm for layer air encapsulation
const std::vector<bool> staveOnFront = makeStaveOnFront(nStaves);
// TODO: fill from the disk optimiser output, see OT_StavePositions above.
const std::vector<std::vector<StaveFill>> exactStaveFills = {};
} // namespace ML_StavePositions

// Get stave configuration based on tracker type
inline StaveConfig getStaveConfig(bool isInnerDisk)
{
  if (isInnerDisk) {
    return StaveConfig{
      true, // isML
      ML_StavePositions::staveID_to_y_midpoint,
      ML_StavePositions::y_lengths,
      ML_StavePositions::x_midpoints,
      ML_StavePositions::x_midpoint_spacing,
      ML_StavePositions::maxToleranceInner,
      ML_StavePositions::maxToleranceOuter,
      ML_StavePositions::staveOnFront,
      ML_StavePositions::exactStaveFills};
  } else {
    return StaveConfig{
      false, // isML
      OT_StavePositions::staveID_to_y_midpoint,
      OT_StavePositions::y_lengths,
      OT_StavePositions::x_midpoints,
      OT_StavePositions::x_midpoint_spacing,
      OT_StavePositions::maxToleranceInner,
      OT_StavePositions::maxToleranceOuter,
      OT_StavePositions::staveOnFront,
      OT_StavePositions::exactStaveFills};
  }
}

} // namespace o2::ft3::ModuleConstants

#endif // FT3MODULECONSTANTS_H