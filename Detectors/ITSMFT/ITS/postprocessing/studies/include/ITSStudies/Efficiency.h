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
o2::framework::DataProcessorSpec getEfficiencyStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC, std::shared_ptr<o2::steer::MCKinematicsReader> kineReader, bool itsStag);

float mEtaCuts[2] = {-1.0, 1.0};
float mPtCuts[2] = {0, 10}; /// no cut for B=0

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

/// excluding bad chips in MC that are not present in data: to be checked based on the anchoring
std::vector<int> mExcludedChipMC = {66, 67, 68, 75, 76, 77, 84, 85, 86, 93, 94, 95, 102, 103, 104, 265, 266, 267, 274, 275, 276, 283, 284, 285, 413, 414, 415, 422, 423, 424, 431, 432, 433};

} // namespace study
} // namespace its
} // namespace o2
#endif