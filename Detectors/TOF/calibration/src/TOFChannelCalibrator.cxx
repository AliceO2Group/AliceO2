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
#include "MathUtils/MathBase.h"
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
using o2::math_utils::math_base::fitGaus;
using boost::histogram::indexed;
  //using boost::histogram::algorithm; // not sure why it does not work...

//_____________________________________________
void TOFChannelData::fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data)
{
  // fill container
  for (int i = data.size(); i--;) {
    auto dt = data[i].getDeltaTimePi();
    auto ch = data[i].getTOFChIndex();
    int sector = ch / o2::tof::Geo::NPADSXSECTOR;
    int chInSect = ch % o2::tof::Geo::NPADSXSECTOR;
    mHisto[sector](dt, chInSect);
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
}
  
//_____________________________________________
  bool TOFChannelData::hasEnoughData(int minEntries) const
{
  // true if all channels can be fitted --> have enough statistics
  
  // we can simply check if the min of the elements of the mEntries vector is >= minEntries
  auto minElementIndex = std::min_element(mEntries.begin(), mEntries.end());
  LOG(INFO) << "minElement is at position " << std::distance(mEntries.begin(), minElementIndex) <<
    " and is " << *minElementIndex;
  bool enough = *minElementIndex < minEntries ? false : true;
  LOG(INFO) << "hasEnough = " << (int)enough; 
  LOG(INFO) << "previous channel has " << *(minElementIndex-1) << " entries"; 
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
    LOG(INFO) << "Number of entries in histogram: " << boost::histogram::algorithm::sum(mHisto[isect]);
    int cnt = 0;
    for (auto&& x : indexed(mHisto[isect])) { // does not work also when I use indexed(*(mHisto[sector]))
      cnt++;
      if (x.get() > 0) {
	LOG(INFO) << "x = " << x.get() << " c " << cnt;
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
    //LOG(INFO) << " c " << cnt << " i " << x.index(0) << " j " << x.index(1) << " b0 " <<  x.bin(0) <<  " b1 " <<  x.bin(1) << " val= " << *x << "|" << x.get(); 
    if (x.get() > 0) {
      LOG(INFO) << "x = " << x.get() << " c " << cnt;
    }
  }
  LOG(INFO) << cnt << " bins inspected";
}

//_____________________________________________
int TOFChannelData::findBin(float v) const
{
  // find the bin along the x-axis (with t-texp) where the value "v" is; this does not depend on the channel
  // (axis 1), nor on the sector, so we use sector0

  if (v == mRange) v -= 1.e-1; 
  
  /*
  LOG(INFO) << "In FindBin, v = : " << v;
  LOG(INFO) << "bin0 limits: lower = " << mHisto[0].axis(0).bin(0).lower() << ", upper = " << mHisto[0].axis(0).bin(0).upper();
  LOG(INFO) << "bin1000 limits: lower = " << mHisto[0].axis(0).bin(mNBins-1).lower() << ", upper = " << mHisto[0].axis(0).bin(mNBins-1).upper();
  LOG(INFO) << "v = " << v << " is in bin " << mHisto[0].axis(0).index(v);
  */  
  return mHisto[0].axis(0).index(v);
  
}

//_____________________________________________
float TOFChannelData::integral(int chmin, int chmax, float binmin, float binmax) const
{
  // calculates the integral in [chmin, chmax] and in [binmin, binmax]

  if (binmin < -mRange || binmax > mRange || chmin < 0 || chmax >= o2::tof::Geo::NCHANNELS)
    throw std::runtime_error("Check your bins, we cannot calculate the integrals in under/overflows bins");
  if (binmax < binmin || chmax < chmin)
    throw std::runtime_error("Check your bin limits!");
  
  int sector = chmin / o2::tof::Geo::NPADSXSECTOR;
  if (sector != chmax / o2::tof::Geo::NPADSXSECTOR) throw std::runtime_error("We cannot integrate over channels that belong to different sectors");

  int chinsectormin = chmin % o2::tof::Geo::NPADSXSECTOR;
  int chinsectormax = chmax % o2::tof::Geo::NPADSXSECTOR;
  //LOG(INFO) << "Calculating integral for channel " << ich << " which is in sector " << sector
  //	    << " (channel in sector is " << chinsector << ")";;
  //  LOG(INFO) << "Bin min = " << binmin << ", binmax = " << binmax <<
  //             ", chmin = " << chmin << ", chmax = " << chmax <<
  //             ", chinsectormin = " << chinsectormin << ", chinsector max = " << chinsectormax;

  float res2 = 0;
  TStopwatch t3;
  int ind = -1;
  int binxmin = findBin(binmin);
  int binxmax = findBin(binmax);
  LOG(DEBUG) << "binxmin = " << binxmin << ", binxmax = " << binxmax;
  t3.Start();
  for (unsigned j = chinsectormin; j <= chinsectormax; ++j) {
    for (unsigned i = binxmin; i <= binxmax; ++i) {
      const auto& v = mHisto[sector].at(i, j);
      res2 += v;
    }
  } 
  t3.Stop();
  LOG(INFO) << "Time for integral looping over axis (result = " << res2 << "):";
  t3.Print();
  
  /* // what is below is only for alternative methods which all proved to be slower
  float res = 0, res1 = 0;
  TStopwatch t1, t2, 
  int startCount = chinsectormin * mNBins + binxmin;
  int endCount =  chinsectormax * mNBins + binxmax; // = startCount + (chinsectormax - chinsectormin) * mNBins + (binxmax - binxmin);
  LOG(DEBUG) << "startCount = " << startCount << " endCount = " << endCount-1;
  t2.Start();
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
  t2.Stop();
  LOG(INFO) << "Time for integral looping over restricted range (result = " << res1 << "):";
  t2.Print();
  t1.Start();
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
  t1.Stop();
  LOG(INFO) << "Time for integral looping (result = " << res << "):";
  t1.Print();
  LOG(INFO) << "Reducing... ";
  TStopwatch t;
  t.Start();
  if (binmin == binmax) binmax += 1.e-1;
  float chinsectorminfl = float(chinsectormin);
  float chinsectormaxfl = float(chinsectormax);
  chinsectormaxfl += 1.e-1; // we need to add a bit because the upper value otherwise is not included
  LOG(DEBUG) << "chinsectorminfl = " << chinsectorminfl << ", chinsectormaxfl = " << chinsectormaxfl << ", binmin= " << binmin << ", binmax = " << binmax;
  LOG(DEBUG) << "chinsectormin = " << chinsectormin << ", chinsectormax = " << chinsectormax;
  auto hch = boost::histogram::algorithm::reduce(mHisto[sector],
  						 boost::histogram::algorithm::shrink(1, chinsectorminfl, chinsectormaxfl),
  						 boost::histogram::algorithm::shrink(0, binmin, binmax)); 
  t.Stop();
  LOG(INFO) << "Time for projection with shrink";
  t.Print();
  LOG(INFO) << "...done.";
  
  //int sizeBeforeAxis1 = mHisto[sector].axis(1).size();
  //int sizeAfterAxis1 = hch.axis(1).size();
  //int sizeBeforeAxis0 = mHisto[sector].axis(0).size();
  //int sizeAfterAxis0 = hch.axis(0).size();
  //std::cout << "axis size before reduction: axis 0: " << sizeBeforeAxis0 << ", axis 1: " << sizeBeforeAxis1 << std::endl;
  //std::cout << "axis size after reduction:  axis 0: " << sizeAfterAxis0 << ", axis 1: " << sizeAfterAxis1 << std::endl;
  
  t.Start();
  auto indhch = indexed(hch);
  const double enthchInd = std::accumulate(indhch.begin(), indhch.end(), 0.0); 
  t.Stop();
  LOG(INFO) << "Time for accumulate (result = " << enthchInd << ")";
  t.Print();

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
  if (binxmax < binxmin || chmax < chmin)
    throw std::runtime_error("Check your bin limits!");

  int sector = chmin / o2::tof::Geo::NPADSXSECTOR;
  if (sector != chmax / o2::tof::Geo::NPADSXSECTOR) throw std::runtime_error("We cannot integrate over channels that belong to different sectors");

  int chinsectormin = chmin % o2::tof::Geo::NPADSXSECTOR;
  int chinsectormax = chmax % o2::tof::Geo::NPADSXSECTOR;
  //LOG(INFO) << "Calculating integral for channel " << ich << " which is in sector " << sector
  //	    << " (channel in sector is " << chinsector << ")";;
  //LOG(INFO) << "Bin min = " << binmin << ", binmax = " << binmax <<
  //             ", chmin = " << chmin << ", chmax" << chmax <<
  //             ", chinsectormin = " << chinsector min << ", chinsector max = " << chinsectormax;
  float res2 = 0;
  TStopwatch t3;
  t3.Start();
  for (unsigned j = chinsectormin; j <= chinsectormax; ++j) {
    for (unsigned i = binxmin; i <= binxmax; ++i) {
      const auto& v = mHisto[sector].at(i, j);
      res2 += v;
    }
  } 
  t3.Stop();
  LOG(INFO) << "Time for integral looping over axis (result = " << res2 << "):";
  t3.Print();
  return res2;
  
  /* // all that is below is alternative methods, all proved to be slower
  float res = 0, res1 = 0;
  TStopwatch t1, t2;
  int ind = -1;
  int startCount = chinsectormin * mNBins + binxmin;
  int endCount =  chinsectormax * mNBins + binxmax; // = startCount + (chinsectormax - chinsectormin) * mNBins + (binxmax - binxmin);
  LOG(DEBUG) << "startCount = " << startCount << " endCount = " << endCount-1;
  t2.Start();
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
  t2.Stop();
  LOG(INFO) << "Time for integral looping over restricted range (result = " << res1 << "):";
  t2.Print();
  t1.Start();
  for (auto&& x : indexed(mHisto[sector])) { 
    ind++;
    if ((x.index(0) >= binxmin && x.index(0) <= binxmax) && (x.index(1) >= chinsectormin && x.index(1) <= chinsectormax)) {
      res += x.get();
      //LOG(INFO) << "ind = " << ind << " will add bin " << x.index(0) << " along x and bin " << x.index(1) << " along y";
    }
  }
  t1.Stop();
  LOG(INFO) << "Time for integral looping (result = " << res << "):";
  t1.Print();
  LOG(INFO) << "Reducing... ";
  TStopwatch t;
  t.Start();
  auto hch = boost::histogram::algorithm::reduce(mHisto[sector],
  						 boost::histogram::algorithm::slice(1, chinsectormin, chinsectormax+1),
  						 boost::histogram::algorithm::slice(0, binxmin, binxmax+1)); // we need to add "+1" 
  t.Stop();
  LOG(INFO) << "Time for projection with slice";
  t.Print();
  //LOG(INFO) << "...done.";

  //int sizeBeforeAxis1 = mHisto[sector].axis(1).size();
  //int sizeAfterAxis1 = hch.axis(1).size();
  //int sizeBeforeAxis0 = mHisto[sector].axis(0).size();
  //int sizeAfterAxis0 = hch.axis(0).size();
  //std::cout << "axis size before reduction: axis 0: " << sizeBeforeAxis0 << ", axis 1: " << sizeBeforeAxis1 << std::endl;
  //std::cout << "axis size after reduction:  axis 0: " << sizeAfterAxis0 << ", axis 1: " << sizeAfterAxis1 << std::endl;
  
  // first way: using indexed (which excludes under/overflow)
  t.Start();
  auto indhch = indexed(hch);
  const double enthchInd = std::accumulate(indhch.begin(), indhch.end(), 0.0); 
  t.Stop();
  LOG(INFO) << "Time for accumulate (result = " << enthchInd << ")";
  t.Print();
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

  return integral(ch, ch, 0, mNBins-1);
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
  return c->hasEnoughData(mMinEntries);
  
}
  
//_____________________________________________
void TOFChannelCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  o2::tof::TOFChannelData* c = slot.getContainer();
  LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  // for the CCDB entry
  std::map<std::string, std::string> md;
  TimeSlewing ts;
  
  for (int ich = 0; ich < o2::tof::Geo::NCHANNELS; ich++) {
    // make the slice of the 2D histogram so that we have the 1D of the current channel
    int sector = ich / o2::tof::Geo::NPADSXSECTOR;
    int chinsector = ich % o2::tof::Geo::NPADSXSECTOR;
    std::vector<float> fitValues;
    std::vector<float> histoValues;
    // reduction is very slow
    //auto hch = boost::histogram::algorithm::reduce( c->getHisto(sector), boost::histogram::algorithm::shrink(1, float(ich), float(ich)+0.1));
    //for (auto&& x : indexed(hch)){
    //  histoValues.push_back(x.get());
    //}

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
    // more efficient way
    auto histo = c->getHisto(sector);
    for (unsigned j = chinsector; j <= chinsector; ++j) {
      for (unsigned i = 0; i < c->getNbins(); ++i) {
	const auto& v = histo.at(i, j);
	histoValues.push_back(v);
      }
    }
    
  int fitres = fitGaus(c->getNbins(), histoValues.data(), -(c->getRange()), c->getRange(), fitValues);
    if (fitres >= 0) {
      LOG(INFO) << "Channel " << ich << " :: Fit result " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    }
    else {
      LOG(ERROR) << "Channel " << ich << " :: Fit failed with result = " << fitres;
    }
    float fractionUnderPeak;
    float intmin = fitValues[1] - 5 * fitValues[2]; // mean - 5*sigma
    float intmax = fitValues[1] + 5 * fitValues[2]; // mean + 5*sigma
    
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
    fractionUnderPeak = c->integral(ich, intmin, intmax) / c->integral(ich);
    // now we need to store the results in the TimeSlewingObject
    ts.setFractionUnderPeak(ich / o2::tof::Geo::NPADSXSECTOR, ich % o2::tof::Geo::NPADSXSECTOR, fractionUnderPeak);
    ts.setSigmaPeak(ich / o2::tof::Geo::NPADSXSECTOR, ich % o2::tof::Geo::NPADSXSECTOR, abs(fitValues[2]));
    ts.addTimeSlewingInfo(ich, 0, fitValues[1]);
  }
  auto clName = o2::utils::MemFileHelper::getClassName(ts);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfoVector.emplace_back("TOF/ChannelCalib", clName, flName, md, slot.getTFStart(), 99999999999999);
  mTimeSlewingVector.emplace_back(ts);

  slot.print();
}

//_____________________________________________
Slot& TOFChannelCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<TOFChannelData>(mNBins, mRange));
  return slot;
}

} // end namespace tof
} // end namespace o2
  
