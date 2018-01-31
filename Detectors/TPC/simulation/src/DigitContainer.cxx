// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitContainer.cxx
/// \brief Implementation of the Digit Container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitContainer.h"
#include <iostream>
#include <cassert>

using namespace o2::TPC;

void DigitContainer::setUp(const short sector, const TimeBin timeBinEvent) {
  mSectorID = sector;
  mSector[getBufferPosition(getSectorLeft(sector))].init(getSectorLeft(sector), timeBinEvent);
  mSector[getBufferPosition(sector)].init(sector, timeBinEvent);
  mSector[getBufferPosition(getSectorRight(sector))].init(getSectorRight(sector), timeBinEvent);
}

unsigned short DigitContainer::getSectorLeft(const short sector)
{
  const int modulus = sector % 18;
  int offsetSector = static_cast<int>(sector / 18) * 18;
  if (modulus == 0)
    offsetSector += 18;
  return offsetSector + modulus - 1;
}

unsigned short DigitContainer::getSectorRight(const short sector)
{
  const int modulus = sector % 18;
  int offsetSector = static_cast<int>(sector / 18) * 18;
  if (modulus == 17)
    offsetSector -= 18;
  return offsetSector + modulus + 1;
}

bool DigitContainer::checkNeighboursProcessed(const short sector) const
{
  return (mSectorProcessed[sector] && mSectorProcessed[getSectorRight(sector)] && mSectorProcessed[getSectorLeft(sector)]);
}

unsigned short DigitContainer::getBufferPosition(const short sector)
{
  if(mSectorMapping[sector] > -1) return mSectorMapping[sector];
  else {
    mSectorMapping[sector] = mNextFreePosition;
    std::cout << "mapping sector " << sector << " to buffer " << mNextFreePosition << "\n";
    //mNextFreePosition = (mNextFreePosition + 1) % 5;
    //return mNextFreePosition;
    return mNextFreePosition++;
  }
}

void DigitContainer::addDigit(size_t eventID, size_t hitID, const CRU &cru, TimeBin timeBin, GlobalPadNumber globalPad, float charge)
{
  const int sector = cru.sector();
  const auto buffer = getBufferPosition(sector);
  assert(0 <= buffer && buffer < 5);
  mSector[getBufferPosition(sector)].setDigit(eventID, hitID, cru, timeBin, globalPad, charge);
}


void DigitContainer::fillOutputContainer(std::vector<Digit> *output, dataformats::MCTruthContainer<MCCompLabel> &mcTruth,
                                         std::vector<DigitMCMetaData> *debug, TimeBin eventTime, bool isContinuous, bool isFinal)
{
  mSectorProcessed[mSectorID] = true;
  for (int buffer = 0; buffer < 5; ++buffer) {
    if (!checkNeighboursProcessed(mSector[buffer].getSector()) && !isFinal) {
      continue;
    }
    std::cout << "writing buffer " << buffer << "\n";
    /// \todo Use CRU here instead of the sector!
    mSector[buffer].fillOutputContainer(output, mcTruth, debug, mSector[buffer].getSector(), eventTime, isContinuous);
    mSectorMapping[mSector[buffer].getSector()] = -1;
    mNextFreePosition = (buffer < mNextFreePosition) ? buffer : mNextFreePosition;
  }
}



