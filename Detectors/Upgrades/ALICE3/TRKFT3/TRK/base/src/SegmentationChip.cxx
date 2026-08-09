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

/// \file SegmentationChip.cxx
/// \brief Implementation of the SegmentationChip class

#include "TRKBase/SegmentationChip.h"
#include <cstdio>

using namespace o2::trk;

// void SegmentationChip::print()
// {
//   printf("++++++++++ VD ++++++++++");
//   printf("Pixel size: %.2f (along %d rows) %.2f (along %d columns) microns\n", PitchRowVD * 1e4, -999, PitchColVD * 1e4, NColsVD);
//   printf("++++++++++ ML ++++++++++");
//   printf("Pixel size: %.2f (along %d rows) %.2f (along %d columns) microns\n", PitchRowML * 1e4, -999, PitchColML * 1e4, NColsML);

// }