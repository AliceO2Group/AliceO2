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

#ifndef FT3MODULECONSTANTS_H
#define FT3MODULECONSTANTS_H

#include <vector>
#include <map>
#include <TColor.h>
#include <TMath.h>

namespace o2::ft3::ModuleConstants
{
/* CURRENT STATUS:
 * 25x32mm sensors, 2mm inactive on one side
 * Most granular layout is 2x1 sensors, where the one on the right has the inactive region
 * on the right, and the one on the left has the inactive region on the left.
 * When stacking 2x1 modules, there is a 0.2mm gap between them. By default, we assume this
 * gap to be ABOVE the most recently placed module.
 *
 * |<- 25mm ->|<- 25mm ->|
 * _______________________
 * -----------------------  0.2mm gap above
 * | |        |        | |
 * | |        |        | |
 * | |        |        | |
 * | |        |        | |  32mm sensor height
 * | |        |        | |
 * | |        |        | |
 * ------------------------
 *
 */
// First set all layout constants for the rest of the function
const double single_sensor_width = 2.5;
const double single_sensor_height = 3.2;
const double inactive_width = 0.2;
const double sensor2x1_gap = 0.02;
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

const int SiColor = kGreen;
const int SiInactiveColor = kRed;
const int glueColor = kBlue;
const int CuColor = kOrange;
const int kaptonColor = kYellow;
const int carbonColor = kBlack;

// Struct for stave position configuration (varies between IT/OT)
struct StaveConfig {
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
  double x_midpoint_spacing;
  // which side of the disc do we place the stave?
  // kSegmentedStave: staggering staves in z (see z_offsetStave)
  // accessed via stave index, NOT stave ID
  const std::vector<bool>& staveOnFront;
};

namespace OT_StavePositions
{
const std::map<int, std::pair<double, bool>> staveID_to_y_midpoint = {
  {-2, {39.0, true}},
  {-1, {41.4, true}},
  {1, {41.4, true}},
  {2, {39.0, true}}};
const std::vector<double> y_lengths = {
  52.8, 66.0, 79.2, 92.4, 99.0, 105.6, 118.8, 118.8,
  128.7, 132.0, 132.0, 138.6, 138.6, 56.1, 52.8,
  52.8, 56.1, 138.6, 138.6, 132.0, 132.0, 128.7,
  118.8, 118.8, 105.6, 99.0, 92.4, 79.2, 66.0, 52.8};
const std::vector<double> x_midpoints = {
  -65.25, -60.75, -56.25, -51.75, -47.25, -42.75, -38.25,       // L
  -33.75, -29.25, -24.75, -20.25, -15.75, -11.25, -6.75, -2.25, // L
  2.25, 6.75, 11.25, 15.75, 20.25, 24.75, 29.25, 33.75,         // R
  38.25, 42.75, 47.25, 51.75, 56.25, 60.75, 65.25               // R
};
const double x_midpoint_spacing = 4.5; // assume constant for now
const std::vector<bool> staveOnFront =
  {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, // L
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0  // R
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
const std::vector<double> x_midpoints = {
  -33.75, -29.25, -24.75, -20.25, -15.75, -11.25, -6.75, -2.25, // L
  2.25, 6.75, 11.25, 15.75, 20.25, 24.75, 29.25, 33.75          // R
};
const double x_midpoint_spacing = 4.5;
const std::vector<bool> staveOnFront =
  {
    1, 0, 1, 0, 1, 0, 1, 0, // L
    0, 1, 0, 1, 0, 1, 0, 1  // R
};
} // namespace ML_StavePositions

// Get stave configuration based on tracker type
inline StaveConfig getStaveConfig(bool isInnerDisk)
{
  if (isInnerDisk) {
    return StaveConfig{
      ML_StavePositions::staveID_to_y_midpoint,
      ML_StavePositions::y_lengths,
      ML_StavePositions::x_midpoints,
      ML_StavePositions::x_midpoint_spacing,
      ML_StavePositions::staveOnFront};
  } else {
    return StaveConfig{
      OT_StavePositions::staveID_to_y_midpoint,
      OT_StavePositions::y_lengths,
      OT_StavePositions::x_midpoints,
      OT_StavePositions::x_midpoint_spacing,
      OT_StavePositions::staveOnFront};
  }
}

} // namespace o2::ft3::ModuleConstants

#endif // FT3MODULECONSTANTS_H