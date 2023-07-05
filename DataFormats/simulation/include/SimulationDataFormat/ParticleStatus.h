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

#ifndef ALICEO2_SIMDATA_PARTICLESTATUS_H_
#define ALICEO2_SIMDATA_PARTICLESTATUS_H_

#include "TParticle.h"

/// enumeration to define status bits for particles in simulation
enum ParticleStatus { kKeep = BIT(14),
                      kDaughters = BIT(15),
                      kToBeDone = BIT(16),
                      kPrimary = BIT(17),
                      kTransport = BIT(18),
                      kInhibited = BIT(19),
                      kHasHits = BIT(20),
                      kHasTrackRefs = BIT(21) };

#endif
