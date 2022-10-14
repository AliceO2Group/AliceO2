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

///
/// @file   FastHisto.h
/// @author Matthias Kleiner, matthias.kleiner@cern.ch
///

#ifndef AliceO2_TPC_FastHisto_H
#define AliceO2_TPC_FastHisto_H

// o2 includes
#include "MathUtils/fit.h"
#include "Framework/Logger.h"
#include <sstream>
#include <vector>
#include <iomanip>
#include "Rtypes.h"

namespace o2
{
namespace tpc
{

/// \brief templated 1D-histogram class.
///
/// This is a templated histogram class without any ROOT includes.
///
/// How to use:
/// e.g.: create an array with 7 bins + underflow bin + overflow bins
/// bin range is from xmin=0.4f to xmax=34.6f
/// the values which will be filled are of the type float
/// #define FMT_HEADER_ONLY; // to avoid undefined reference when using root shel
/// o2::tpc::FastHisto<float> histo(7,0.4f,34.6f,true,true);
/// histo.fill(4.5);
/// histo.fill(20.2,3);
/// histo.fill(35);
/// histo.print()
/// float mean = histo.getStatisticsData(0,1).mCOG;
/// float TruncatedMean = histo.getStatisticsData(0.05,.6).mCOG;

template <class T>
class FastHisto
{

 public:
  /// constructor
  /// this function is called when initializing the histogram object
  /// \param nBins number of bins used in the histogram
  /// \param xmin minimum x value
  /// \param xmax maximum x value (value is not included in the histogram. same as TH1 in ROOT)
  /// \param useUnderflow use underflow bin in the histogram
  /// \param useOverflow use overflow bin in the histogram
  FastHisto(const unsigned int nBins = 20, const float xmin = 0.f, const float xmax = 2.f, const bool useUnderflow = true, const bool useOverflow = true)
    : mNBins(nBins), mXmin(xmin), mXmax(xmax), mUseUnderflow(useUnderflow), mUseOverflow(useOverflow), mBinCont(nBins + useUnderflow + useOverflow), mBinWidth((mXmax - mXmin) / mNBins){};

  /// default destructor
  ~FastHisto() = default;

  /// this function fills the histogram with given value and weight
  /// \param val value which will be filled in the histogram
  /// \param weight the weight of the filled value
  void fill(const float val, T weight = 1);

  /// this function returns the index (bin) for the input value
  /// \return the index of the bin
  /// \param val the value for which the bin index will be returned
  int findBin(const float val) const;

  /// this function resets the bin content in the histogram
  void reset()
  {
    mBinCount = 0;
    std::fill(mBinCont.begin(), mBinCont.end(), 0);
  }

  /// this function increase the bin content for given index with given weight
  /// \param index the index (bin) for which the bin content is increased
  /// \param weight weight of the filled content
  void fillBin(int index, T weight)
  {
    ++mBinCount;
    mBinCont[index] += weight;
  }

  /// this function prints out the histogram
  /// \param type printing type e.g 'vertical printing: type=0', 'horizontal printing: type=1'
  /// \param prec sets the precision of the x axis label
  // TODO function prints only integers, add to print histgramm with 0.x values
  void print(const int prec = 2) const;

  /// \return this function returns the truncated mean for the filled histogram
  /// \param low lower truncation range
  /// \param high upper truncation range
  math_utils::StatisticsData getStatisticsData(const float low = 0.05f, const float high = 0.6f) const;

  /// \return this function returns the bin content for given index
  /// \param index the index (bin) for which the content is returned
  T getBinContent(unsigned int index) const
  {
    if (checkBin(index)) {
      return mBinCont[index];
    } else {
      return -1;
    }
  }

  /// \return returns the center of the bin
  /// \param bin bin
  float getBinCenter(int bin) const
  {
    const float binWidth = getBinWidth();
    const float val = (bin - mUseUnderflow) * binWidth + mXmin;
    return val;
  }

  /// \return the bin width used in the histogram
  float getBinWidth() const { return mBinWidth; }

  /// \return returns the number of bins (excluding underflow/overflow)
  unsigned int getNBins() const { return mNBins; }

  /// \return get the lower bound of the histogram
  float getXmin() const { return mXmin; }

  /// \return get the upper bound of the histogram
  float getXmax() const { return mXmax; }

  /// \return return number of entries in the histogram
  unsigned int getEntries() const { return mBinCount; }

  /// \return returns if underflow bin is used in the histogram
  bool isUnderflowSet() const { return mUseUnderflow; }

  /// \return returns if overflow bin is used in the histogram
  bool isOverflowSet() const { return mUseOverflow; }

  /// \return returns status wether the bin is in the histogram range
  bool checkBin(int bin) const
  {
    unsigned int vecsize = mBinCont.size();
    if (bin >= vecsize) {
      return 0;
    } else {
      return 1;
    }
  }

  /// overload of operator +
  const FastHisto& operator+=(const FastHisto& other);

 private:
  unsigned int mNBins{};     ///< number of bins used
  float mXmin{};             ///< minimum x value in the histogram
  float mXmax{};             ///< maximum x value in the histogram (value not included)
  bool mUseUnderflow = true; ///< if true underflow bin used in the histogram
  bool mUseOverflow = true;  ///< if true overflow bin is used in the histogram
  float mBinWidth{};         ///< width of the bins
  std::vector<T> mBinCont{}; ///< histogram containing bin content
  unsigned int mBinCount{0}; ///< number of values which are filled in the histogram

  ClassDefNV(FastHisto, 1)
};

//______________________________________________________________________________
template <class T>
inline void FastHisto<T>::fill(const float val, T weight)
{
  const int indexBin = findBin(val);
  if (indexBin == -1) { // if no underflow/overflow bin is used, but the value should be in the underflow/overflow bin return
    return;
  }
  fillBin(indexBin, weight); // fill the correct index with given weight
}

template <class T>
inline void FastHisto<T>::print(const int prec) const
{
  const math_utils::StatisticsData data = getStatisticsData();
  LOGP(info, "\n Entries: {}", mBinCount);
  LOGP(info, "Truncated Mean: {}", data.mCOG);
  LOGP(info, "Standard Deviation: {}", data.mStdDev);
  LOGP(info, "sum of content: {}", data.mSum);

  const int maxEle = *std::max_element(mBinCont.begin(), mBinCont.end()); // get the maximum element in the
                                                                          // histogram needed for printing
  // this loop prints the histogram
  // starting from the largest value in the array and go backwards to 0
  std::stringstream stream;
  stream.width(10);

  // getting the x axis label
  stream << "  1";
  for (int i = 2; i <= maxEle; ++i) {
    stream.width(2);
    stream << std::right << i;
  }
  LOGP(info, "{}", stream.str());
  stream.str(std::string());
  stream << "---------";
  for (int i = 0; i <= maxEle; ++i) {
    stream << "--";
  }
  LOGP(info, "{}", stream.str());

  const float binWidth = getBinWidth();
  const int totalBins = mBinCont.size();

  for (int i = 0; i < totalBins; ++i) {
    stream.str(std::string());
    if (i == 0 && mUseUnderflow) {
      stream.width(9);
      stream << std::right << "U_Flow | ";
    } else if (i == totalBins - 1 && mUseOverflow) {
      stream.width(9);
      stream << std::right << "O_Flow | ";
    } else {
      stream.width(6);
      const float xPos = mXmin + binWidth * (i - mUseUnderflow);
      stream << std::fixed << std::setprecision(prec) << xPos;
      const std::string sIn = " | ";
      stream << sIn;
    }

    for (int j = 1; j <= getBinContent(i); ++j) {
      const std::string sIn = "x ";
      stream << sIn;
    }
    std::string xPosString = stream.str();
    LOGP(info, "{}", xPosString);
  }
  LOGP(info, "");
}

template <class T>
inline math_utils::StatisticsData FastHisto<T>::getStatisticsData(const float low, const float high) const
{
  math_utils::StatisticsData data{};
  // in case something went wrong the COG is the histogram lower limit
  data.mCOG = mXmin;
  if (mBinCont.size() == 0) {
    return data;
  }
  if (low > high) {
    return data;
  }

  // total histogram content
  const T integral = std::accumulate(mBinCont.begin(), mBinCont.end(), 0);
  if (integral == 0) {
    return data;
  }
  const float lowerBound = integral * low;
  const float upperBound = integral * high;
  const unsigned int lastBin = mBinCont.size() - 1;

  float mean = 0;
  float sum = 0;
  float rms2 = 0;

  // add fractional values
  const float binWidth = getBinWidth();
  const int shiftBin = mUseUnderflow ? -1 : 0; // shift the bin position in case of additional bin on the left

  T countContent = 0;
  bool isLowerFrac = false;
  unsigned int bin = 0;
  for (bin = 0; bin <= lastBin; ++bin) {
    countContent += mBinCont[bin];

    if (countContent - lowerBound <= 0) { // lower truncation cut
      continue;
    } else if (countContent >= upperBound) { // upper truncation cut
      break;
    }

    const float xcenter = mXmin + (bin + 0.5f + shiftBin) * binWidth;
    const float tmpBinCont = isLowerFrac ? mBinCont[bin] : countContent - lowerBound; // set bincontent to countcontent for first time only
    const float tmpMean = tmpBinCont * xcenter;
    mean += tmpMean;
    sum += tmpBinCont;
    rms2 += tmpMean * xcenter;
    isLowerFrac = true;
  }

  if (!checkBin(bin)) {
    return data;
  }

  // set last bin
  // TODO move to upper loop
  const float xcenter = mXmin + (bin + 0.5f + shiftBin) * binWidth;
  const T upFrac = mBinCont[bin] - (static_cast<float>(countContent) - upperBound);
  const float tmpMean = upFrac * xcenter;
  mean += tmpMean;
  sum += upFrac;
  rms2 += tmpMean * xcenter;

  if (sum == 0) {
    return data;
  }

  mean /= sum;
  data.mCOG = mean;
  rms2 /= sum;
  data.mStdDev = std::sqrt(std::abs(rms2 - mean * mean));
  data.mSum = sum;
  return data;
}

template <class T>
inline int FastHisto<T>::findBin(const float val) const
{
  if (val < mXmin) {
    if (!mUseUnderflow) {
      LOGP(warning, "findBin: UNDERFLOW BIN: BIN IS NOT IN HISTOGRAM RANGE!");
      return -1; // if undeflow bin is not used BUT value is in underflow bin return -1
    } else {
      return 0; // if underflow bin is used return 0 as index
    }
  }

  if (val >= mXmax) {
    if (!mUseOverflow) {
      LOGP(warning, "findBin: OVERFLOW BIN: BIN IS NOT IN HISTOGRAM RANGE!");
      return -1; // if overflow bin is not used BUT value is in overflow bin return -1
    } else {
      return mBinCont.size() - 1; // if overflow bin is used return the last index in the vector
    }
  }

  const float binWidth = getBinWidth();
  const int bin = (val - mXmin) / binWidth + mUseUnderflow;
  return bin;
};

template <class T>
inline const FastHisto<T>& FastHisto<T>::operator+=(const FastHisto& other)
{
  if (other.mBinCount == 0) {
    return *this;
  }

  // make sure the calibration objects have the same substructure
  if (mNBins != other.mNBins || mXmin != other.mXmin || mXmax != other.mXmax || mUseUnderflow != other.mUseUnderflow || mUseOverflow != other.mUseOverflow) {
    static int errCount = 0;
    if (mBinCount && errCount < 10) {
      errCount++;
      LOGP(warning, "mBinCount {} other.mBinCount: {} mNBins {}, other.mNBins {}, mXmin {}, other.mXmin {}, mXmax {}, other.mXmax {}, mUseUnderflow {}, other.mUseUnderflow {}, mUseOverflow {}, other.mUseOverflow {}", mBinCount, other.mBinCount, mNBins, other.mNBins, mXmin, other.mXmin, mXmax, other.mXmax, mUseUnderflow, other.mUseUnderflow, mUseOverflow, other.mUseOverflow);
    }
    *this = other;
    return *this;
  }
  mBinCount += other.mBinCount;
  std::transform(mBinCont.begin(), mBinCont.end(), other.mBinCont.begin(), mBinCont.begin(), std::plus<T>());
  return *this;
}

} // namespace tpc
} // namespace o2

#endif
