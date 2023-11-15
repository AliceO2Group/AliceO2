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
#include <boost/histogram/algorithm/sum.hpp>
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
  stream << "EMCAL Cell ID:  " << mTimeHisto[0] << "\n";
}
//_____________________________________________
void EMCALTimeCalibData::print()
{
  LOG(debug) << *this;
}
//_____________________________________________
std::ostream& operator<<(std::ostream& stream, const EMCALTimeCalibData& emcdata)
{
  emcdata.PrintStream(stream);
  return stream;
}
//_____________________________________________
void EMCALTimeCalibData::merge(EMCALTimeCalibData* prev)
{
  mEvents += prev->getNEvents();
  mNEntriesInHisto += prev->getNEntriesInHisto();
  mTimeHisto[0] += prev->getHisto();
}
//_____________________________________________
bool EMCALTimeCalibData::hasEnoughData() const
{
  bool enough = false;

  LOG(debug) << "mNEntriesInHisto: " << mNEntriesInHisto << " needed: " << EMCALCalibParams::Instance().minNEntries_tc << "  mEvents = " << mEvents;
  // use enrties in histogram for calibration
  if (!EMCALCalibParams::Instance().useNEventsForCalib_tc && mNEntriesInHisto > EMCALCalibParams::Instance().minNEntries_tc) {
    enough = true;
  }
  // use number of events (from emcal trigger record) for calibration
  if (EMCALCalibParams::Instance().useNEventsForCalib_tc && mEvents > EMCALCalibParams::Instance().minNEvents_tc) {
    enough = true;
  }

  return enough;
}

//_____________________________________________
void EMCALTimeCalibData::fill(const gsl::span<const o2::emcal::Cell> data)
{
  // the fill function is called once per event
  mEvents++;

  if (data.size() == 0) {
    return;
  }
  auto fillfunction = [this](int thread, const gsl::span<const o2::emcal::Cell> data, double minCellEnergy) {
    LOG(debug) << "filling in thread " << thread << " ncells = " << data.size();
    auto& mCurrentHist = mTimeHisto[thread];
    unsigned int nEntries = 0;
    for (auto cell : data) {
      double cellEnergy = cell.getEnergy();
      double cellTime = cell.getTimeStamp();
      int id = cell.getTower();
      if (mApplyGainCalib) {
        LOG(debug) << " gain calib factor for cell " << id << " = " << mGainCalibFactors->getGainCalibFactors(id);
        cellEnergy *= mGainCalibFactors->getGainCalibFactors(id);
      }
      if (cellEnergy > minCellEnergy) {
        LOG(debug) << "inserting in cell ID " << id << ": cellTime = " << cellTime;
        mCurrentHist(cellTime, id);
        nEntries++;
      }
    }
    mVecNEntriesInHisto[thread] += nEntries;
  };

  std::vector<gsl::span<const o2::emcal::Cell>> ranges(mNThreads);
  auto size_per_thread = static_cast<unsigned int>(std::ceil((static_cast<float>(data.size()) / mNThreads)));
  unsigned int currentfirst = 0;
  for (int ithread = 0; ithread < mNThreads; ithread++) {
    unsigned int nelements = std::min(size_per_thread, static_cast<unsigned int>(data.size() - 1 - currentfirst));
    ranges[ithread] = data.subspan(currentfirst, nelements);
    currentfirst += nelements;
    LOG(debug) << "currentfirst " << currentfirst << "  nelements " << nelements;
  }

  double minCellEnergy = EMCALCalibParams::Instance().minCellEnergy_tc;

#if (defined(WITH_OPENMP) && !defined(__CLING__))
  LOG(debug) << "Number of threads that will be used = " << mNThreads;
#pragma omp parallel for num_threads(mNThreads)
#else
  LOG(debug) << "OPEN MP will not be used for the bad channel calibration";
#endif
  for (int ithread = 0; ithread < mNThreads; ithread++) {
    fillfunction(ithread, ranges[ithread], minCellEnergy);
  }

  // only sum up entries if needed
  if (!o2::emcal::EMCALCalibParams::Instance().useNEventsForCalib_tc) {
    for (auto& nEntr : mVecNEntriesInHisto) {
      mNEntriesInHisto += nEntr;
      nEntr = 0;
    }
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
