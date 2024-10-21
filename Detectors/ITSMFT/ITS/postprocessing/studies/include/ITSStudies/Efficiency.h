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

#ifndef O2_EFFICIENCY_STUDY_H
#define O2_EFFICIENCY_STUDY_H

#include "Framework/DataProcessorSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

namespace o2
{
namespace steer
{
class MCKinematicsReader;
}
namespace its
{
namespace study
{
using mask_t = o2::dataformats::GlobalTrackID::mask_t;
o2::framework::DataProcessorSpec getEfficiencyStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC, std::shared_ptr<o2::steer::MCKinematicsReader> kineReader);

////// phi cuts for B=0
float mPhiCutsL0[10][2] = {{-122.5, -122}, {-91.8, -91.7}, {-61, -60}, {-30.1, -29.8}, {30, 30.2}, {59, 59.5}, {88, 89}, {117, 118.5}, {147, 147.8}, {176.5, 176.6}};
float mPhiCutsL1[12][2] = {{-137, -136.5}, {-114, -113.5}, {-91.5, -91}, {-68.5, -68}, {-45.6, -45.4}, {-23.1, -22.9}, {45.4, 45.6}, {67.4, 67.6}, {89.4, 89.6}, {110.4, 110.6}, {132.4, 132.6}, {154.4, 154.6}};
float mPhiCutsL2[17][2] = {{-162.85, -162.65}, {-145, -144.5}, {-127, -126.5}, {-109, -108.5}, {-91, -90.5}, {-73, -72.5}, {-55.1, -54.9}, {-37.35, -37.15}, {-19.5, -19}, {36.8, 37}, {54.4, 54.6}, {71.9, 72.1}, {89, 89.5}, {106.4, 106.6}, {123.65, 123.85}, {141.4, 141.6}, {158.9, 159.1}};

float mEtaCuts[2] = {-1.0, 1.0};
// float mPtCuts[2] = {1, 4.5}; //// for B=5
float mPtCuts[2] = {0, 10}; /// no cut for B=0
int mChi2cut = 100;

// values obtained from the dca study for B=5
// float dcaXY[3] = {-0.000326, -0.000217, -0.000187};
// float dcaZ[3] = {0.000020, -0.000004, 0.000032};
// float sigmaDcaXY[3] = {0.001375, 0.001279, 0.002681};
// float sigmaDcaZ[3] = {0.002196, 0.002083, 0.004125};

// values obtained from the dca study for B=0
float dcaXY[3] = {-0.000328, -0.000213, -0.000203};
float dcaZ[3] = {-0.000000543, -0.000013, 0.000001};
float sigmaDcaXY[3] = {0.00109, 0.000895, 0.001520};
float sigmaDcaZ[3] = {0.001366, 0.001149, 0.001868};

int dcaCut = 8;

float mDCACutsXY[3][2] = {{dcaXY[0] - dcaCut * sigmaDcaXY[0], dcaXY[0] + dcaCut* sigmaDcaXY[0]}, {dcaXY[1] - dcaCut * sigmaDcaXY[1], dcaXY[1] + dcaCut* sigmaDcaXY[1]}, {dcaXY[2] - dcaCut * sigmaDcaXY[2], dcaXY[2] + dcaCut* sigmaDcaXY[2]}}; // cuts at 8 sigma for each layer for xy. The values represent m-8sigma and m+8sigma
float mDCACutsZ[3][2] = {{dcaZ[0] - dcaCut * sigmaDcaZ[0], dcaZ[0] + dcaCut* sigmaDcaZ[0]}, {dcaZ[1] - dcaCut * sigmaDcaZ[1], dcaZ[1] + dcaCut* sigmaDcaZ[1]}, {dcaZ[2] - dcaCut * sigmaDcaZ[2], dcaZ[2] + dcaCut* sigmaDcaZ[2]}};

} // namespace study
} // namespace its
} // namespace o2
#endif