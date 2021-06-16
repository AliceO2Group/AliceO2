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

/// @file  VtxTrackIndex.h
/// \brief Extention of GlobalTrackID by flags relevant for verter-track association
/// \author ruben.shahoyan@cern.ch

#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "Framework/Logger.h"
#include <fmt/printf.h>
#include <iostream>
#include <bitset>

using namespace o2::dataformats;
