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

///
/// \file Digitizer.cxx
/// \brief Implementation of the ALICE3 TOF digitizer
/// \author Nicolò Jacazio, Università del Piemonte Orientale (IT)
/// \since 2026-03-17
///

#include "IOTOFSimulation/Digitizer.h"
#include "DetectorsRaw/HBFUtils.h"

#include <TRandom.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <fairlogger/Logger.h>

namespace o2::iotof
{

void Digitizer::init()
{
}

void Digitizer::process(const std::vector<o2::itsmft::Hit>* hits, int evID, int srcID)
{
}

void Digitizer::processHit(const o2::itsmft::Hit& hit, uint32_t& maxFr, int evID, int srcID)
{
}

} // namespace o2::iotof
