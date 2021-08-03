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

#include "EMCALCalibration/EMCALTimeCalibData.h"
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

using Slot = o2::calibration::TimeSlot<o2::emcal::EMCALTimeCalibData>;
using clbUtils = o2::calibration::Utils;
using boost::histogram::indexed;

//===================================================================
//_____________________________________________
void EMCALTimeCalibData::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Cell ID:  " << mTimeHisto << "\n";
}
//_____________________________________________
void EMCALTimeCalibData::print()
{
  LOG(DEBUG) << *this;
}
//_____________________________________________
std::ostream& operator<<(std::ostream& stream, const EMCALTimeCalibData& emcdata)
{
  emcdata.PrintStream(stream);
  return stream;
}
//_____________________________________________
void EMCALTimeCalibData::merge(const EMCALTimeCalibData* prev)
{
  mEvents += prev->getNEvents();
  mTimeHisto += prev->getHisto();
}
//_____________________________________________
bool EMCALTimeCalibData::hasEnoughData() const
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
//_____________________________________________
void EMCALTimeCalibData::fill(const gsl::span<const o2::emcal::Cell> data)
{
  for (auto cell : data) {
    Double_t cellEnergy = cell.getEnergy();
    Int_t id = cell.getTower();
    LOG(DEBUG) << "inserting in cell ID " << id << ": energy = " << cellEnergy;
    mTimeHisto(cellEnergy, id);
  }
}

//_____________________________________________
o2::emcal::TimeCalibrationParams EMCALTimeCalibData::process()
{
  o2::emcal::TimeCalibrationParams TimeCalibParams;
  return TimeCalibParams;
}

} // end namespace emcal
} // end namespace o2
