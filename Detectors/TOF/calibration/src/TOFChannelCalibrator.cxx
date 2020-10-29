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
#include "MathUtils/fit.h"
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
namespace tof
{

using Slot = o2::calibration::TimeSlot<o2::tof::TOFChannelData>;
using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;
using clbUtils = o2::calibration::Utils;
using boost::histogram::indexed;
using o2::math_utils::fitGaus;
//using boost::histogram::algorithm; // not sure why it does not work...

//_____________________________________________
void TOFChannelData::fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data)
{
  // fill container
  for (int i = data.size(); i--;) {
    auto ch = data[i].getTOFChIndex();
    int sector = ch / o2::tof::Geo::NPADSXSECTOR;
    int chInSect = ch % o2::tof::Geo::NPADSXSECTOR;
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
void TOFChannelData::merge(const TOFChannelData* prev)
{
  // merge data of 2 slots
  for (int isect = 0; isect < o2::tof::Geo::NSECTORS; isect++) {
    mHisto[isect] += prev->getHisto(isect);
  }
  for (int ich = 0; ich < o2::tof::Geo::NCHANNELS; ich++) {
    mEntries[ich] += prev->mEntries[ich];
  }
}

//_____________________________________________
bool TOFChannelData::hasEnoughData(int minEntries) const
{
  // true if all channels can be fitted --> have enough statistics

  // TODO: think of a better logic: it could be that some channels are off or misbehaving and do not
  // have enough entries, even if in principle we collected enough

  // we can simply check if the min of the elements of the mEntries vector is >= minEntries
  printEntries();
  auto minElementIndex = std::min_element(mEntries.begin(), mEntries.end());
  LOG(INFO) << "minElement is at position " << std::distance(mEntries.begin(), minElementIndex) << " and is " << *minElementIndex;
  bool enough = *minElementIndex < minEntries ? false : true;
  LOG(INFO) << "hasEnough = (minEntry=" << minEntries << ") " << (int)enough;
  return enough;
}

//_____________________________________________
void TOFChannelData::print() const
{
  LOG(INFO) << "Printing histograms:";
  std::ostringstream os;
  for (int isect = 0; isect < o2::tof::Geo::NSECTORS; isect++) {
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
  for (int i = 0; i < mEntries.size(); ++i)
    LOG(INFO) << "channel " << i << " has " << mEntries[i] << " entries";
}

//_____________________________________________
int TOFChannelData::findBin(float v) const
{
  // find the bin along the x-axis (with t-texp) where the value "v" is; this does not depend on the channel
  // (axis 1), nor on the sector, so we use sector0

  if (v == mRange)
    v -= 1.e-1;

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

  if (binmin < -mRange || binmax > mRange || chmin < 0 || chmax >= o2::tof::Geo::NCHANNELS) {
    throw std::runtime_error("Check your bins, we cannot calculate the integrals in under/overflows bins");
  }
  if (binmax < binmin || chmax < chmin) {
    throw std::runtime_error("Check your bin limits!");
  }

  int sector = chmin / o2::tof::Geo::NPADSXSECTOR;
  if (sector != chmax / o2::tof::Geo::NPADSXSECTOR)
    throw std::runtime_error("We cannot integrate over channels that belong to different sectors");

  int chinsectormin = chmin % o2::tof::Geo::NPADSXSECTOR;
  int chinsectormax = chmax % o2::tof::Geo::NPADSXSECTOR;

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

  /* // what is below is only for alternative methods which all proved to be slower
  float res = 0, res1 = 0;
  //TStopwatch t1, t2, 
  int startCount = chinsectormin * mNBins + binxmin;
  int endCount =  chinsectormax * mNBins + binxmax; // = startCount + (chinsectormax - chinsectormin) * mNBins + (binxmax - binxmin);
  LOG(DEBUG) << "startCount = " << startCount << " endCount = " << endCount-1;
  //t2.Start();
  int counts = -1;
  for (auto&& x : indexed(mHisto[sector])) {
    counts++;
    if (counts < startCount) continue;
    if (x.bin(0).lower() > binmax && chinsectormin == chinsectormax) { // all others also will be > but only if chmin = chmax; in the other cases, we should jump to the next row,which for now we cannot do in boost
      //      LOG(INFO) << "x.bin(0).lower() > binmax && chinsectormin == chinsectormax: BREAKING";
      break;
    }
    if (x.index(1) > chinsectormax) { // we passed the needed channel
      //LOG(INFO) << "x.index(1) > chinsectormax: BREAKING";
      break;
    }
    if ( (x.bin(0).upper() > binmin) && (x.bin(0).lower() <= binmax) && (x.index(1) >= chinsectormin)) { // I have to keep the condition "&& (x.bin(0).lower() <= binmax)" because I can break only if chmin == chmax
      res1 += x.get();
      //if (x.get() != 0) LOG(INFO) << "ind = " << counts << " will add bin " << x.index(0)
      //				  << " along x (in [" << x.bin(0).lower() << ", "
      //				  << x.bin(0).upper() << "], and bin " << x.index(1) << " along y" << " with content " << x.get() << " --> res1 = " << res1;
    }
  }
  //t2.Stop();
  //LOG(DEBUG) << "Time for integral looping over restricted range (result = " << res1 << "):";
  //t2.Print();
  //t1.Start();
  ind = -1;
  for (auto&& x : indexed(mHisto[sector])) { 
    ind++;
    if ((x.bin(0).upper() > binmin && x.bin(0).lower() < binmax) && (x.index(1) >= chinsectormin && x.index(1) <= chinsectormax)) {
      res += x.get();
      //if (x.get() != 0) LOG(INFO) << "ind = " << ind << " will add bin " << x.index(0)
      //				  << " along x (in [" << x.bin(0).lower() << ", "
      //			  << x.bin(0).upper() << "], and bin " << x.index(1) << " along y" << " with content " << x.get();
    }
  }
  //t1.Stop();
  //LOG(DEBUG) << "Time for integral looping (result = " << res << "):";
  //t1.Print();
  LOG(DEBUG) << "Reducing... ";
  //TStopwatch t;
  //t.Start();
  if (binmin == binmax) binmax += 1.e-1;
  float chinsectorminfl = float(chinsectormin);
  float chinsectormaxfl = float(chinsectormax);
  chinsectormaxfl += 1.e-1; // we need to add a bit because the upper value otherwise is not included
  LOG(DEBUG) << "chinsectorminfl = " << chinsectorminfl << ", chinsectormaxfl = " << chinsectormaxfl << ", binmin= " << binmin << ", binmax = " << binmax;
  LOG(DEBUG) << "chinsectormin = " << chinsectormin << ", chinsectormax = " << chinsectormax;
  auto hch = boost::histogram::algorithm::reduce(mHisto[sector],
  						 boost::histogram::algorithm::shrink(1, chinsectorminfl, chinsectormaxfl),
  						 boost::histogram::algorithm::shrink(0, binmin, binmax)); 
  //t.Stop();
  //LOG(DEBUG) << "Time for projection with shrink";
  //t.Print();
  //LOG(DEBUG) << "...done.";
  
  //int sizeBeforeAxis1 = mHisto[sector].axis(1).size();
  //int sizeAfterAxis1 = hch.axis(1).size();
  //int sizeBeforeAxis0 = mHisto[sector].axis(0).size();
  //int sizeAfterAxis0 = hch.axis(0).size();
  //std::cout << "axis size before reduction: axis 0: " << sizeBeforeAxis0 << ", axis 1: " << sizeBeforeAxis1 << std::endl;
  //std::cout << "axis size after reduction:  axis 0: " << sizeAfterAxis0 << ", axis 1: " << sizeAfterAxis1 << std::endl;
  
  //t.Start();
  auto indhch = indexed(hch);
  const double enthchInd = std::accumulate(indhch.begin(), indhch.end(), 0.0); 
  //t.Stop();
  //LOG(DEBUG) << "Time for accumulate (result = " << enthchInd << ")";
  //t.Print();

  return enthchInd;
*/
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

  if (binxmin < 0 || binxmax > mNBins || chmin < 0 || chmax >= o2::tof::Geo::NCHANNELS)
    throw std::runtime_error("Check your bins, we cannot calculate the integrals in under/overflows bins");
  if (binxmax < binxmin || chmax < chmin) {
    throw std::runtime_error("Check your bin limits!");
  }

  int sector = chmin / o2::tof::Geo::NPADSXSECTOR;
  if (sector != chmax / o2::tof::Geo::NPADSXSECTOR)
    throw std::runtime_error("We cannot integrate over channels that belong to different sectors");

  int chinsectormin = chmin % o2::tof::Geo::NPADSXSECTOR;
  int chinsectormax = chmax % o2::tof::Geo::NPADSXSECTOR;

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

  /* // all that is below is alternative methods, all proved to be slower
  float res = 0, res1 = 0;
  //TStopwatch t1, t2;
  int ind = -1;
  int startCount = chinsectormin * mNBins + binxmin;
  int endCount =  chinsectormax * mNBins + binxmax; // = startCount + (chinsectormax - chinsectormin) * mNBins + (binxmax - binxmin);
  LOG(DEBUG) << "startCount = " << startCount << " endCount = " << endCount-1;
  //t2.Start();
  int counts = -1;
  for (auto&& x : indexed(mHisto[sector])) {
    counts++;
    if (counts < startCount) continue;
    if (x.index(0) > binxmax && chinsectormin == chinsectormax) { // all others also will be > but only if chmin = chmax; in the other cases, we should jump to the next row,which for now we cannot do in boost
      //LOG(INFO) << "x.index(0) > binxmax && chinsectormin == chinsectormax: BREAKING";
      break;
    }
    if (x.index(1) > chinsectormax) { // we passed the needed channel
      //LOG(INFO) << "x.index(1) > chinsectormax) > chinsectormax: BREAKING";
      break;
    }
    if ( (x.index(0) >= binxmin) && (x.index(0) <= binxmax) && (x.index(1) >= chinsectormin)) { // I have to keep the condition "&& (x.bin(0).lower() <= binmax)" because I can break only if chmin == chmax
	res1 += x.get();
	//	if (x.get() != 0) 
	// LOG(INFO) << "ind = " << counts << " will add bin " << x.index(0)
	//	    << " along x (in [" << x.bin(0).lower() << ", "
	//	    << x.bin(0).upper() << "], and bin " << x.index(1) << " along y" << " with content " << x.get()
	//	    << " --> res1 = " << res1;
    }
  }
  //t2.Stop();
  //LOG(DEBUG) << "Time for integral looping over restricted range (result = " << res1 << "):";
  //t2.Print();
  //t1.Start();
  for (auto&& x : indexed(mHisto[sector])) { 
    ind++;
    if ((x.index(0) >= binxmin && x.index(0) <= binxmax) && (x.index(1) >= chinsectormin && x.index(1) <= chinsectormax)) {
      res += x.get();
      //LOG(INFO) << "ind = " << ind << " will add bin " << x.index(0) << " along x and bin " << x.index(1) << " along y";
    }
  }
  //t1.Stop();
  //LOG(DEBUG) << "Time for integral looping (result = " << res << "):";
  //t1.Print();
  //LOG(DEBUG) << "Reducing... ";
  //TStopwatch t;
  //t.Start();
  auto hch = boost::histogram::algorithm::reduce(mHisto[sector],
  						 boost::histogram::algorithm::slice(1, chinsectormin, chinsectormax+1),
  						 boost::histogram::algorithm::slice(0, binxmin, binxmax+1)); // we need to add "+1" 
  //t.Stop();
  //LOG(DEBUG) << "Time for projection with slice";
  //t.Print();
  //LOG(INFO) << "...done.";

  //int sizeBeforeAxis1 = mHisto[sector].axis(1).size();
  //int sizeAfterAxis1 = hch.axis(1).size();
  //int sizeBeforeAxis0 = mHisto[sector].axis(0).size();
  //int sizeAfterAxis0 = hch.axis(0).size();
  //std::cout << "axis size before reduction: axis 0: " << sizeBeforeAxis0 << ", axis 1: " << sizeBeforeAxis1 << std::endl;
  //std::cout << "axis size after reduction:  axis 0: " << sizeAfterAxis0 << ", axis 1: " << sizeAfterAxis1 << std::endl;
  
  // first way: using indexed (which excludes under/overflow)
  //t.Start();
  auto indhch = indexed(hch);
  const double enthchInd = std::accumulate(indhch.begin(), indhch.end(), 0.0); 
  //t.Stop();
  //LOG(DEBUG) << "Time for accumulate (result = " << enthchInd << ")";
  //t.Print();
  return enthchInd;
  */
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

//===================================================================

//_____________________________________________
void TOFChannelCalibrator::initOutput()
{
  // Here we initialize the vector of our output objects
  mInfoVector.clear();
  mTimeSlewingVector.clear();
  return;
}

//_____________________________________________
bool TOFChannelCalibrator::hasEnoughData(const Slot& slot) const
{

  // Checking if all channels have enough data to do calibration.
  // Delegating this to TOFChannelData

  const o2::tof::TOFChannelData* c = slot.getContainer();
  LOG(INFO) << "Checking statistics";
  return (mTest ? true : c->hasEnoughData(mMinEntries));
}

//_____________________________________________
void TOFChannelCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  o2::tof::TOFChannelData* c = slot.getContainer();
  LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  // for the CCDB entry
  std::map<std::string, std::string> md;
  TimeSlewing& ts = mCalibTOFapi->getSlewParamObj(); // we take the current CCDB object, since we want to simply update the offset

  for (int ich = 0; ich < o2::tof::Geo::NCHANNELS; ich++) {
    // make the slice of the 2D histogram so that we have the 1D of the current channel
    int sector = ich / o2::tof::Geo::NPADSXSECTOR;
    int chinsector = ich % o2::tof::Geo::NPADSXSECTOR;
    std::vector<float> fitValues;
    std::vector<float> histoValues;
    /* //less efficient way
    int startCount = chinsector * c->getNbins();
    int counts = -1;
    for (auto&& x : indexed(c->getHisto(sector))) {
      counts++;
      if (counts < startCount) continue;
      if (x.index(1) > chinsector) { // we passed the needed channel
	//LOG(INFO) << "x.index(1) > chinsectormax) > chinsectormax: BREAKING";
	break;
      }
      histoValues.push_back(x.get());
    }
    */
    std::vector<int> entriesPerChannel = c->getEntriesPerChannel();
    if (entriesPerChannel.at(ich) == 0) {
      continue; // skip always since a channel with 0 entries is normal, it will be flagged as problematic
      if (mTest) {
        LOG(DEBUG) << "Skipping channel " << ich << " because it has zero entries, but it should not be"; // should become error!
        continue;
      } else
        throw std::runtime_error("We found one channel with no entries, we cannot calibrate!");
    }

    // more efficient way
    auto histo = c->getHisto(sector);
    for (unsigned j = chinsector; j <= chinsector; ++j) {
      for (unsigned i = 0; i < c->getNbins(); ++i) {
        const auto& v = histo.at(i, j);
        LOG(DEBUG) << "channel = " << ich << ", in sector = " << sector << " (where it is channel = " << chinsector << ") bin = " << i << " value = " << v;
        histoValues.push_back(v);
      }
    }

    double fitres = fitGaus(c->getNbins(), histoValues.data(), -(c->getRange()), c->getRange(), fitValues);

    if (fitValues[2] < 0)
      fitValues[2] = -fitValues[2];

    if (fitres >= 0) {
      LOG(DEBUG) << "Channel " << ich << " :: Fit result " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(INFO) << "Channel " << ich << " :: Fit failed with result = " << fitres;
    }
    float fractionUnderPeak;
    float intmin = fitValues[1] - 5 * fitValues[2]; // mean - 5*sigma
    float intmax = fitValues[1] + 5 * fitValues[2]; // mean + 5*sigma

    if (intmin < -mRange)
      intmin = -mRange;
    if (intmax < -mRange)
      intmax = -mRange;
    if (intmin > mRange)
      intmin = mRange;
    if (intmax > mRange)
      intmax = mRange;

    /* 
    // needed if we calculate the integral using the values
    int binmin = c->findBin(intmin);
    int binmax = c->findBin(intmax);

    // for now these checks are useless, as we pass the value of the bin
    if (binmin < 0)
      binmin = 1; // avoid to take the underflow bin (can happen in case the sigma is too large)
    if (binmax >= c->getNbins())
      binmax = c->getNbins()-1; // avoid to take the overflow bin (can happen in case the sigma is too large)
    float fractionUnderPeak = (c->integral(ch, binmin, binmax) + addduetoperiodicity) / c->integral(ch, 1, c->nbins());
    */
    float channelIntegral = c->integral(ich);
    fractionUnderPeak = channelIntegral > 0 ? c->integral(ich, intmin, intmax) / channelIntegral : 0;
    // now we need to store the results in the TimeSlewingObject
    ts.setFractionUnderPeak(ich / o2::tof::Geo::NPADSXSECTOR, ich % o2::tof::Geo::NPADSXSECTOR, fractionUnderPeak);
    ts.setSigmaPeak(ich / o2::tof::Geo::NPADSXSECTOR, ich % o2::tof::Geo::NPADSXSECTOR, abs(fitValues[2]));
    ts.updateOffsetInfo(ich, fitValues[1]);
  }
  auto clName = o2::utils::MemFileHelper::getClassName(ts);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfoVector.emplace_back("TOF/ChannelCalib", clName, flName, md, slot.getTFStart(), 99999999999999);
  mTimeSlewingVector.emplace_back(ts);
}

//_____________________________________________
Slot& TOFChannelCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<TOFChannelData>(mNBins, mRange, mCalibTOFapi));
  return slot;
}

} // end namespace tof
} // end namespace o2
