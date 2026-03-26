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

  const double active_width = single_sensor_width - inactive_width;
  const double active_height = single_sensor_height;

  const double sensor2x1_width = 2 * single_sensor_width;
  const double sensor2x1_active_width = 2 * active_width;
  const double sensor2x1_height = single_sensor_height;
  const unsigned kSensorsPerStack = 1;
  const double sensor_stack_height = kSensorsPerStack * sensor2x1_height +
                                    (kSensorsPerStack - 1) * sensor2x1_gap;

  const double carbonFiberThickness = 0.01;
  const double foamSpacingThickness = 1.0;

  /* 
   * Constants for staves are written for both positive
   * and negative x even though they are just mirrored now,
   * because there might be design changes in the future
   * that require a non-mirrored layout, making it easier to
   * change here if so required, even though it looks uglier now.
   */ 
  // First define midpoints of staves that would overlap with inner disc
  // EXCEPTION: Assumed mirrored around x-axis
  // map from Stave ID (1-indexed from other documents) to midpoint
  // Do NOT add any zero midpoints, this is taken off separately
  const std::map<int, double> staveID_to_y_midpoint = {
    {-2, 39.0},
    {-1, 41.4},
    {1, 41.4},
    {2, 39.0}
  };
  // lengths of staves, their midpoint, and their face
  const std::vector<double> y_lengths = {
    52.8, 66.0, 79.2, 92.4, 99.0, 105.6, 118.8, 118.8,
    128.7, 132.0, 132.0, 138.6, 138.6, 56.1, 52.8,
    52.8, 56.1, 138.6, 138.6, 132.0, 132.0, 128.7,
    118.8, 118.8, 105.6, 99.0, 92.4, 79.2, 66.0, 52.8
  };
  const std::vector<double> x_midpoints = {
    -65.25, -60.75, -56.25, -51.75, -47.25, -42.75, -38.25,  // L
    -33.75, -29.25, -24.75, -20.25, -15.75, -11.25, -6.75, -2.25,  // L
    2.25, 6.75, 11.25, 15.75, 20.25, 24.75, 29.25, 33.75,  // R
    38.25, 42.75, 47.25, 51.75, 56.25, 60.75, 65.25  // R
  };
  // which side of the disc do we place the stave?
  // accessed via stave index, NOT stave ID
  const std::vector<bool> staveOnFront =
  {
     1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  // L
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0  // R
  };

  // small helper function to get 1-indexed stave ID, counting from the middle outwards,
  // with negative IDs on the left and positive IDs on the right
  inline const int staveIdxToID(int staveIdx) {
    unsigned nStavesOneSide = staveOnFront.size() / 2;
    bool isRight = staveIdx >= nStavesOneSide;
    return staveIdx - nStavesOneSide + isRight;
  }

  // material properties
  const double siliconThickness = 0.01;
  const double copperThickness = 0.006;
  const double kaptonThickness = 0.03;
  const double epoxyThickness = 0.0012;

  const int SiColor = kGreen;
  const int SiInactiveColor = kRed;
  const int glueColor = kBlue;
  const int CuColor = kBlack;
  const int kaptonColor = kYellow;
}

#endif // FT3MODULECONSTANTS_H