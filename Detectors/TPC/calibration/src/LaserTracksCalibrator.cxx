// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Logger.h"

#include "TPCCalibration/LaserTracksCalibrator.h"

using namespace o2::tpc;

void LaserTracksCalibrator::initOutput()
{
  mDVperSlot.clear();
}

//______________________________________________________________________________
void LaserTracksCalibrator::finalizeSlot(Slot& slot)
{
  auto& calibLaser = *slot.getContainer();
  calibLaser.finalize();
  calibLaser.print();

  mDVperSlot.emplace_back(calibLaser.getDVall());
}

//______________________________________________________________________________
LaserTracksCalibrator::Slot& LaserTracksCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<CalibLaserTracks>());
  return slot;
}
