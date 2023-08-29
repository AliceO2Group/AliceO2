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
#if (defined(WITH_OPENMP) && !defined(__CLING__))
#include <omp.h>
#endif

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
  stream << "EMCAL Cell ID:  " << mHisto[0] << "\n";
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
  // the fill function is called once per event
  mEvents++;

  if (data.size() == 0) {
    return;
  }

  auto fillfunction = [this](int thread, const gsl::span<const o2::emcal::Cell>& data, double minCellEnergy, double minCellEnergyTime) {
    LOG(debug) << "filling in thread " << thread << " ncells = " << data.size();
    auto& mCurrentHist = mHisto[thread];
    auto& mCurrentHistTime = mHistoTime[thread];
    unsigned int nEntries = 0; // counter set inside function increases speed compared to simply using mVecNEntriesInHisto[thread]. Added to global counter at end of function
    for (const auto& cell : data) {
      int id = cell.getTower();
      double cellEnergy = cell.getEnergy();

      if (mApplyGainCalib) {
        LOG(debug) << " gain calib factor for cell " << id << " = " << mArrGainCalibFactors[id];
        cellEnergy *= mArrGainCalibFactors[id];
      }

      if (cellEnergy < minCellEnergy) {
        LOG(debug) << "skipping cell ID " << id << ": with energy = " << cellEnergy << " below  threshold of " << minCellEnergy;
        continue;
      }

      LOG(debug) << "inserting in cell ID " << id << ": energy = " << cellEnergy;
      mCurrentHist(cellEnergy, id);
      nEntries++;

      if (cellEnergy > minCellEnergyTime) {
        double cellTime = cell.getTimeStamp();
        LOG(debug) << "inserting in cell ID " << id << ": time = " << cellTime;
        mCurrentHistTime(cellTime, id);
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
  }

  double minCellEnergy = o2::emcal::EMCALCalibParams::Instance().minCellEnergy_bc;
  double minCellEnergyTime = o2::emcal::EMCALCalibParams::Instance().minCellEnergyTime_bc;

#if (defined(WITH_OPENMP) && !defined(__CLING__))
  LOG(info) << "Number of threads that will be used = " << mNThreads;
#pragma omp parallel for num_threads(mNThreads)
#else
  LOG(info) << "OPEN MP will not be used for the bad channel calibration";
#endif
  for (int ithread = 0; ithread < mNThreads; ithread++) {
    fillfunction(ithread, ranges[ithread], minCellEnergy, minCellEnergyTime);
  }

  // only sum up entries if needed
  if (!o2::emcal::EMCALCalibParams::Instance().useNEventsForCalib_bc) {
    for (auto& nEntr : mVecNEntriesInHisto) {
      mNEntriesInHisto += nEntr;
      nEntr = 0;
    }
  }
}
//_____________________________________________
void EMCALChannelData::print()
{
  LOG(debug) << *this;
}
//_____________________________________________
void EMCALChannelData::merge(EMCALChannelData* prev)
{
  mEvents += prev->getNEvents();
  mNEntriesInHisto += prev->getNEntriesInHisto();
  mHisto[0] += prev->getHisto();
  mHistoTime[0] += prev->getHisto();
}

//_____________________________________________
bool EMCALChannelData::hasEnoughData() const
{
  bool enough = false;

  LOG(debug) << "mNEntriesInHisto: " << mNEntriesInHisto * mNThreads << " needed: " << EMCALCalibParams::Instance().minNEntries_bc << "  mEvents = " << mEvents;
  // use entries in histogram for calibration
  if (!EMCALCalibParams::Instance().useNEventsForCalib_bc) {
    long unsigned int nEntries = 0;
    if (mNEntriesInHisto > EMCALCalibParams::Instance().minNEntries_bc) {
      enough = true;
    }
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
  mOutputBCM = mCalibExtractor->calibrateBadChannels(mEsumHisto, getHistoTime());
}
//____________________________________________

} // end namespace emcal
} // end namespace o2