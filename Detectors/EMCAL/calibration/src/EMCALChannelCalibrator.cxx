// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALCalibration/EMCALChannelCalibrator.h"
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

//_____________________________________________
void EMCALChannelCalibrator::initOutput()
{
  mInfoVector.clear();
  return;
}

//_____________________________________________
bool EMCALChannelCalibrator::hasEnoughData(const Slot& slot) const
{

  const o2::emcal::EMCALChannelData* c = slot.getContainer();
  LOG(INFO) << "Checking statistics";
  return (mTest ? true : c->hasEnoughData());
}

//_____________________________________________
void EMCALChannelCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  o2::emcal::EMCALChannelData* c = slot.getContainer();
  LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  // for the CCDB entry
  std::map<std::string, std::string> md;

  //auto clName = o2::utils::MemFileHelper::getClassName(tm);
  //auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfoVector.emplace_back("EMCAL/ChannelCalib", "clname", "flname", md, slot.getTFStart(), 99999999999999);
}

//_____________________________________________
Slot& EMCALChannelCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<o2::emcal::EMCALChannelData>(mNBins, mRange));
  return slot;
}

} // end namespace emcal
} // end namespace o2
