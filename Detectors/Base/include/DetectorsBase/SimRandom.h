// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_BASE_SIMRANDOM_H
#define ALICEO2_BASE_SIMRANDOM_H

#include <TRandom.h>

namespace o2::base
{

/// The random number generator used during particle transport, for seeding and for drawing.
/// Each thread can install its own generator; a thread that has none uses gRandom.
TRandom* simRandom();

/// Install the transport generator of the calling thread (not owned); nullptr reverts to gRandom.
void setSimRandom(TRandom* random);

} // namespace o2::base

#endif
