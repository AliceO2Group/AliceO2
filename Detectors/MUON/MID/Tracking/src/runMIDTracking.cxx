// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   runMIDClusterizer.cxx
/// \brief  A simple program to reconstruct MID tracks
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 May 2017

#include "runFairMQDevice.h"
#include "TrackerDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options) {}

FairMQDevicePtr getDevice(const FairMQProgOptions&) { return new o2::mid::TrackerDevice(); }
