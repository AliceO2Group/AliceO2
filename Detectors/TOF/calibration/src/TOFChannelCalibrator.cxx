// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFCalibration/TOFChannelCalibrator.h"
#include "Framework/Logger.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <TStopwatch.h>

namespace o2
{
namespace tof
{

using Slot = o2::calibration::TimeSlot<o2::tof::TOFChannelData>;
using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;
using clbUtils = o2::calibration::Utils;
using boost::histogram::indexed;
using namespace o2::tof;
//using boost::histogram::algorithm; // not sure why it does not work...

//_____________________________________________
void TOFChannelData::fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data)
{
  // fill container
  for (int i = data.size(); i--;) {
    auto ch = data[i].getTOFChIndex();
    int sector = ch / Geo::NPADSXSECTOR;
    int chInSect = ch % Geo::NPADSXSECTOR;
    auto dt = data[i].getDeltaTimePi();

    auto tot = data[i].getTot();
    // TO BE DISCUSSED: could it be that the LHCphase is too old? If we ar ein sync mode, it could be that it is not yet created for the current run, so the one from the previous run (which could be very old) is used. But maybe it does not matter much, since soon enough a calibrated LHC phase should be produced
    auto corr = mCalibTOFapi->getTimeCalibration(ch, tot); // we take into account LHCphase, offsets and time slewing
    LOG(DEBUG) << "inserting in channel " << ch << ": dt = " << dt << ", tot = " << tot << ", corr = " << corr << ", corrected dt = " << dt - corr;

    dt -= corr;

    mHisto[sector](dt, chInSect); // we pass the calibrated time
    mEntries[ch] += 1;
  }
}

//_____________________________________________
void TOFChannelData::fill(const gsl::span<const o2::tof::CalibInfoCluster> data)
{
  // fill container
  for (int i = data.size(); i--;) {
    auto ch = data[i].getCH();
    auto dch = data[i].getDCH(); // this is a char! if you print it, you need to cast it to int
    auto dt = data[i].getDT();
    auto tot1 = data[i].getTOT1();
    auto tot2 = data[i].getTOT2();

    // we order them so that the channel number of the first cluster is smaller than
    // the one of the second cluster
    if (dch < 0) {
      ch += dch;
      dt = -dt;
      dch = -dch;
      float inv = tot1;
      tot1 = tot2;
      tot2 = inv;
    }

    int sector = ch / Geo::NPADSXSECTOR;
    int absoluteStrip = ch / Geo::NPADS;
    int stripInSect = absoluteStrip % Geo::NSTRIPXSECTOR;
    int shift = 0;
    if (dch == 1) {
      shift = 0; // 2nd channel is on the right
    } else if (dch == 48) {
      shift = 1; // 2nd channel is at the top
    } else {
      continue;
    }
    int chOnStrip = ch % 96;
    int comb = 96 * shift + chOnStrip; // index of the current pair of clusters on the strip; there are in total 96 + 48

    auto corr1 = mCalibTOFapi->getTimeCalibration(ch, tot1);       // we take into account LHCphase, offsets and time slewing
    auto corr2 = mCalibTOFapi->getTimeCalibration(ch + dch, tot2); // we take into account LHCphase, offsets and time slewing
    LOG(DEBUG) << "inserting in channel " << ch << ", " << ch + dch << ": dt = " << dt << ", tot1 = " << tot1 << ", tot2 = " << tot2 << ", corr1 = " << corr1 << ", corr2 = " << corr2 << ", corrected dt = " << dt - corr1 + corr2;

    dt -= corr1 - corr2;

    int combInSect = comb + stripInSect * NCOMBINSTRIP;

    LOG(DEBUG) << "ch = " << ch << ", sector = " << sector << ", absoluteStrip = " << absoluteStrip << ", stripInSect = " << stripInSect << ", shift = " << shift << ", dch = " << (int)dch << ", chOnStrip = " << chOnStrip << ", comb = " << comb;

    mHisto[sector](dt, combInSect); // we pass the difference of the *calibrated* times
    mEntries[comb + NCOMBINSTRIP * absoluteStrip] += 1;
  }
}

//_____________________________________________
void TOFChannelData::merge(const TOFChannelData* prev)
{
  // merge data of 2 slots
  for (int isect = 0; isect < Geo::NSECTORS; isect++) {
    mHisto[isect] += prev->getHisto(isect);
  }
  for (auto iel = 0; iel < mEntries.size(); iel++) {
    mEntries[iel] += prev->mEntries[iel];
  }
}

//_____________________________________________
bool TOFChannelData::hasEnoughData(int minEntries) const
{
  // true if all channels can be fitted --> have enough statistics

  // We consider that we have enough entries if the mean of the number of entries in the channels
  // with at least one entry is greater than the cut at "minEntries"
  // Channels/pairs with zero entries are assumed to be off --> we do not consider them

  //printEntries();
  int nValid = 0;
  float mean = 0;
  int smallestElementIndex = -1;
  int smallestEntries = 1e5;
  int largestElementIndex = -1;
  int largestEntries = 0;
  for (auto i = 0; i < mEntries.size(); ++i) {
    if (mEntries[i] != 0) { // skipping channels/pairs if they have zero entries (most likely they are simply off)
      mean += mEntries[i];
      ++nValid;
      if (mEntries[i] < smallestEntries) {
        smallestEntries = mEntries[i];
        smallestElementIndex = i;
      }
      if (mEntries[i] > largestEntries) {
        largestEntries = mEntries[i];
        largestElementIndex = i;
      }
    }
  }
  if (nValid == 0) {
    LOG(INFO) << "hasEnough = false: all channels/pairs are empty";
    return false;
  }

  LOG(DEBUG) << "mean = " << mean << ", nvalid = " << nValid;
  mean /= nValid;

  LOG(DEBUG) << "minElement is at position " << smallestElementIndex << " and has " << smallestEntries << " entries";
  LOG(DEBUG) << "maxElement is at position " << largestElementIndex << " and has " << largestEntries << " entries";
  float threshold = minEntries + 5 * std::sqrt(minEntries);
  bool enough = mean < threshold ? false : true;
  if (enough) {
    LOG(INFO) << "hasEnough: " << (int)enough << " ("
              << nValid << " valid channels found (should be " << mEntries.size() << ") with mean = "
              << mean << " with cut at = " << threshold << ") ";
  }
  return enough;
}

//_____________________________________________
void TOFChannelData::print() const
{
  LOG(INFO) << "Printing histograms:";
  std::ostringstream os;
  for (int isect = 0; isect < Geo::NSECTORS; isect++) {
    LOG(INFO) << "Sector: " << isect;
    os << mHisto[isect];
    auto nentriesInSec = boost::histogram::algorithm::sum(mHisto[isect]);
    LOG(INFO) << "Number of entries in histogram: " << boost::histogram::algorithm::sum(mHisto[isect]);
    int cnt = 0;
    if (nentriesInSec != 0) {
      for (auto&& x : indexed(mHisto[isect])) {
        if (x.get() > 0) {
          const auto i = x.index(0); // current index along first axis --> t-texp
          const auto j = x.index(1); // current index along second axis --> channel
          const auto b0 = x.bin(0);  // current bin interval along first axis --> t-texp
          const auto b1 = x.bin(1);  // current bin interval along second axis --> channel
          LOG(INFO) << "bin " << cnt << ": channel = " << j << " in [" << b1.lower() << ", " << b1.upper()
                    << "], t-texp in [" << b0.lower() << ", " << b0.upper() << "], has entries = " << x.get();
        }
        cnt++;
      }
    }
    LOG(INFO) << cnt << " bins inspected";
  }
}

//_____________________________________________
void TOFChannelData::print(int isect) const
{
  LOG(INFO) << "*** Printing histogram " << isect;
  std::ostringstream os;
  int cnt = 0;
  os << mHisto[isect];
  LOG(INFO) << "Number of entries in histogram: " << boost::histogram::algorithm::sum(mHisto[isect]);
  for (auto&& x : indexed(mHisto[isect])) { // does not work also when I use indexed(*(mHisto[sector]))
    cnt++;
    LOG(DEBUG) << " c " << cnt << " i " << x.index(0) << " j " << x.index(1) << " b0 " << x.bin(0) << " b1 " << x.bin(1) << " val= " << *x << "|" << x.get();
    if (x.get() > 0) {
      LOG(INFO) << "x = " << x.get() << " c " << cnt;
    }
  }
  LOG(INFO) << cnt << " bins inspected";
}

//_____________________________________________
void TOFChannelData::printEntries() const
{
  // to print number of entries per channel
  for (int i = 0; i < mEntries.size(); ++i) {
    if (mEntries.size() > tof::Geo::NCHANNELS) {
      LOG(INFO) << "pair of channels " << i << " has " << mEntries[i] << " entries";
    } else {
      LOG(INFO) << "channel " << i << " has " << mEntries[i] << " entries";
    }
  }
}

//_____________________________________________
int TOFChannelData::findBin(float v) const
{
  // find the bin along the x-axis (with t-texp) where the value "v" is; this does not depend on the channel
  // (axis 1), nor on the sector, so we use sector0

  if (v == mRange) {
    v -= 1.e-1;
  }

  LOG(DEBUG) << "In FindBin, v = : " << v;
  LOG(DEBUG) << "bin0 limits: lower = " << mHisto[0].axis(0).bin(0).lower() << ", upper = " << mHisto[0].axis(0).bin(0).upper();
  LOG(DEBUG) << "bin1000 limits: lower = " << mHisto[0].axis(0).bin(mNBins - 1).lower() << ", upper = " << mHisto[0].axis(0).bin(mNBins - 1).upper();
  LOG(DEBUG) << "v = " << v << " is in bin " << mHisto[0].axis(0).index(v);

  return mHisto[0].axis(0).index(v);
}

//_____________________________________________
float TOFChannelData::integral(int chmin, int chmax, float binmin, float binmax) const
{
  // calculates the integral in [chmin, chmax] and in [binmin, binmax]

  if (binmin < -mRange || binmax > mRange || chmin < 0 || chmax >= Geo::NSECTORS * mNElsPerSector) {
    throw std::runtime_error("Check your bins, we cannot calculate the integrals in under/overflows bins");
  }
  if (binmax < binmin || chmax < chmin) {
    throw std::runtime_error("Check your bin limits!");
  }

  int sector = chmin / mNElsPerSector;
  if (sector != chmax / mNElsPerSector) {
    throw std::runtime_error("We cannot integrate over channels that belong to different sectors");
  }

  int chinsectormin = chmin % mNElsPerSector;
  int chinsectormax = chmax % mNElsPerSector;

  float res2 = 0;
  //TStopwatch t3;
  int ind = -1;
  int binxmin = findBin(binmin);
  int binxmax = findBin(binmax);
  LOG(DEBUG) << "binxmin = " << binxmin << ", binxmax = " << binxmax;
  //t3.Start();
  for (unsigned j = chinsectormin; j <= chinsectormax; ++j) {
    for (unsigned i = binxmin; i <= binxmax; ++i) {
      const auto& v = mHisto[sector].at(i, j);
      res2 += v;
    }
  }
  //t3.Stop();
  LOG(DEBUG) << "Time for integral looping over axis (result = " << res2 << "):";
  //t3.Print();

  return res2;

}

//_____________________________________________
float TOFChannelData::integral(int ch, float binmin, float binmax) const
{
  // calculates the integral along one fixed channel and in [binmin, binmax]

  return integral(ch, ch, binmin, binmax);
}

//_____________________________________________
float TOFChannelData::integral(int chmin, int chmax, int binxmin, int binxmax) const
{
  // calculates the integral in [chmin, chmax] and in [binmin, binmax]

  if (binxmin < 0 || binxmax > mNBins || chmin < 0 || chmax >= Geo::NSECTORS * mNElsPerSector) {
    throw std::runtime_error("Check your bins, we cannot calculate the integrals in under/overflows bins");
  }
  if (binxmax < binxmin || chmax < chmin) {
    throw std::runtime_error("Check your bin limits!");
  }

  int sector = chmin / mNElsPerSector;
  if (sector != chmax / mNElsPerSector) {
    throw std::runtime_error("We cannot integrate over channels that belong to different sectors");
  }

  int chinsectormin = chmin % mNElsPerSector;
  int chinsectormax = chmax % mNElsPerSector;

  float res2 = 0;
  //TStopwatch t3;
  //t3.Start();
  for (unsigned j = chinsectormin; j <= chinsectormax; ++j) {
    for (unsigned i = binxmin; i <= binxmax; ++i) {
      const auto& v = mHisto[sector].at(i, j);
      res2 += v;
    }
  }
  //t3.Stop();
  LOG(DEBUG) << "Time for integral looping over axis (result = " << res2 << "):";
  //t3.Print();
  return res2;

}

//_____________________________________________
float TOFChannelData::integral(int ch, int binxmin, int binxmax) const
{
  // calculates the integral along one fixed channel and in [binmin, binmax]

  return integral(ch, ch, binxmin, binxmax);
}

//_____________________________________________
float TOFChannelData::integral(int ch) const
{
  // calculates the integral along one fixed channel and in the full x-range

  return integral(ch, ch, 0, mNBins - 1);
}

} // end namespace tof
} // end namespace o2
