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

using namespace o2::TPC;

void DigitTime::setDigit(size_t eventID, size_t hitID, const CRU &cru, GlobalPadNumber globalPad, float charge)
{
  mGlobalPads[globalPad].setDigit(eventID, hitID, charge);
  const int gemStack = static_cast<int>(cru.gemStack());
  mCommonMode[gemStack] += charge;
}

void DigitTime::fillOutputContainer(std::vector<Digit> *output, dataformats::MCTruthContainer<MCCompLabel> &mcTruth,
                                    std::vector<DigitMCMetaData> *debug, Sector sector, TimeBin timeBin)
{
  const static Mapper &mapper = Mapper::instance();
  GlobalPadNumber currentPad = 0;
  for(auto &globalPad : mGlobalPads) {
    /// Only actual signals are written to disk
    if(globalPad.getChargePad() >0 ) {
      /// \todo Obtain CRU from globalPadPos and Sector
      const int cru = sector.getSector();
      globalPad.fillOutputContainer(output, mcTruth, debug, cru, timeBin, currentPad, getCommonMode(cru));
    }
    ++currentPad;
  }
}
