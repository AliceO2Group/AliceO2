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
#include "EMCALCalibration/EMCALCalibExtractor.h"
#include "CommonUtils/BoostHistogramUtils.h"
// #include "Framework/Logger.h"
// #include "CommonUtils/MemFileHelper.h"
// #include "CCDB/CcdbApi.h"
// #include "DetectorsCalibration/Utils.h"
// #include <boost/histogram.hpp>
// #include <boost/histogram/ostream.hpp>
// #include <boost/format.hpp>
// #include <cassert>
// #include <iostream>
// #include <sstream>
// #include <TStopwatch.h>

namespace o2
{
namespace emcal
{

using Slot = o2::calibration::TimeSlot<o2::emcal::EMCALChannelData>;
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
  //the fill function is called once per event
  mEvents++;
  for (auto cell : data) {
    double cellEnergy = cell.getEnergy();
    int id = cell.getTower();
    LOG(debug) << "inserting in cell ID " << id << ": energy = " << cellEnergy;
    mHisto(cellEnergy, id);
    mNEntriesInHisto++;

    if (cellEnergy > o2::emcal::EMCALCalibParams::Instance().minCellEnergyTime_bc) {
      double cellTime = cell.getTimeStamp();
      LOG(debug) << "inserting in cell ID " << id << ": time = " << cellTime;
      mHistoTime(cellTime, id);
    }
  }
}
//_____________________________________________
void EMCALChannelData::print()
{
  LOG(debug) << *this;
}
//_____________________________________________
void EMCALChannelData::merge(const EMCALChannelData* prev)
{
  mEvents += prev->getNEvents();
  mNEntriesInHisto += prev->getNEntriesInHisto();
  mHisto += prev->getHisto();
  mHistoTime += prev->getHistoTime();
}

//_____________________________________________
bool EMCALChannelData::hasEnoughData() const
{
  bool enough = false;

  LOG(debug) << "mNEntriesInHisto: " << mNEntriesInHisto << " needed: " << EMCALCalibParams::Instance().minNEntries_bc << "  mEvents = " << mEvents;
  // use enrties in histogram for calibration
  if (!EMCALCalibParams::Instance().useNEventsForCalib_bc && mNEntriesInHisto > EMCALCalibParams::Instance().minNEntries_bc) {
    enough = true;
  }
  // use number of events (from emcal trigger record) for calibration
  if (EMCALCalibParams::Instance().useNEventsForCalib_bc && mEvents > EMCALCalibParams::Instance().minNEvents_bc) {
    enough = true;
  }

  return enough;
}

//_____________________________________________
void EMCALChannelData::analyzeSlot()
{
  mOutputBCM = mCalibExtractor->calibrateBadChannels(mEsumHisto, mHistoTime);
}
//____________________________________________

} // end namespace emcal
} // end namespace o2