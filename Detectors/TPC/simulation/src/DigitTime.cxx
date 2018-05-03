// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitTime.cxx
/// \brief Implementation of the Time Bin container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitTime.h"
#include "TPCBase/Mapper.h"

using namespace o2::TPC;

void DigitTime::addDigit(size_t eventID, size_t trackID, const CRU& cru, GlobalPadNumber globalPad, float signal)
{
  mGlobalPads[globalPad].addDigit(eventID, trackID, signal);
  mCommonMode[cru.gemStack()] += signal;
}

void DigitTime::fillOutputContainer(std::vector<Digit>* output, dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                                    std::vector<DigitMCMetaData>* debug, const Sector& sector, TimeBin timeBin,
                                    float commonMode)
{
  static Mapper& mapper = Mapper::instance();
  GlobalPadNumber globalPad = 0;
  for (auto& pad : mGlobalPads) {
    const int cru = mapper.getCRU(sector, globalPad);
    pad.fillOutputContainer(output, mcTruth, debug, cru, timeBin, globalPad, getCommonMode(cru));
    ++globalPad;
  }
}
