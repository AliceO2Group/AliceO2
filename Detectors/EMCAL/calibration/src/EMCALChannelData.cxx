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

#include "EMCALCalibration/EMCALChannelData.h"
#include "Framework/Logger.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/format.hpp>
#include <cassert>
#include <iostream>
#include <sstream>
#include <TStopwatch.h>

namespace o2
{
namespace emcal
{

using Slot = o2::calibration::TimeSlot<o2::emcal::EMCALChannelData>;
using clbUtils = o2::calibration::Utils;
using boost::histogram::indexed;

//===================================================================
//_____________________________________________
void EMCALChannelData::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Cell ID:  " << mHisto << "\n";
}
//_____________________________________________
std::ostream& operator<<(std::ostream& stream, const EMCALChannelData& emcdata)
{
  emcdata.PrintStream(stream);
  return stream;
}
//_____________________________________________
void EMCALChannelData::fill(const gsl::span<const o2::emcal::Cell> data)
{
  for (auto cell : data) {
    Double_t cellEnergy = cell.getEnergy();
    Int_t id = cell.getTower();
    LOG(DEBUG) << "inserting in cell ID " << id << ": energy = " << cellEnergy;
    mHisto(cellEnergy, id);
  }
}
//_____________________________________________
void EMCALChannelData::print()
{
  LOG(DEBUG) << *this;
}
//_____________________________________________
void EMCALChannelData::merge(const EMCALChannelData* prev)
{
  mEvents += prev->getNEvents();
  mHisto += prev->getHisto();
}

//_____________________________________________
bool EMCALChannelData::hasEnoughData() const
{
  // true if we have enough data, also want to check for the sync trigger
  // this is stil to be finalized, simply a skeletron for now

  // if we have the sync trigger, finalize the slot anyway

  //finalizeOldestSlot(Slot& slot);

  // TODO: use event counter here to specify the value of enough
  // guess and then adjust number of events as needed
  // checking mEvents
  bool enough;

  return enough;
}

} // end namespace emcal
} // end namespace o2