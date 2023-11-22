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

/// \file ROFTimeClusterFinder.h
/// \brief Class to group the fired pads according to their time stamp
///
/// \author Andrea Ferrero, CEA

#ifndef O2_MCH_ROFTIMECLUSTERFINDERV2_H_
#define O2_MCH_ROFTIMECLUSTERFINDERV2_H_

#include <cassert>
#include <cstdint>
#include <vector>
#include <gsl/span>

#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "MCHDigitFiltering/DigitFilter.h"

namespace o2
{
namespace mch
{

class ROFTimeClusterFinderV2
{
 public:
  using ROFVector = std::vector<o2::mch::ROFRecord>;

  ROFTimeClusterFinderV2(gsl::span<const o2::mch::ROFRecord> rofs,
                         gsl::span<const o2::mch::Digit> digits,
                         uint32_t timeClusterSize,
                         uint32_t peakSearchWindow,
                         uint32_t nBins,
                         uint32_t nDigitsMin,
                         bool improvePeakSearch,
                         bool mergeROFs,
                         bool debug);
  ~ROFTimeClusterFinderV2() = default;

  /// process the digit ROFs and create the time clusters
  void process();

  /// return the vector of time-cluster ROFs
  const ROFVector& getROFRecords() const { return mOutputROFs; }

  /// stores the output ROFs into a flat memory buffer
  char* saveROFRsToBuffer(size_t& bufSize);

  /// debugging output
  void dumpInputROFs();
  void dumpOutputROFs();

 private:
  /// structure representing one bin in the time histogram used for the peak search algorithm
  struct TimeBin {
    int32_t mFirstIdx{-1};  ///< index of the first digit ROF in the bin
    int32_t mLastIdx{-1};   ///< index of the last digit ROF in the bin
    uint32_t mNDigitsPS{0}; ///< number of digits in the bin for the peak search step

    TimeBin() = default;

    bool empty() { return mNDigitsPS == 0; }

    bool operator<(const TimeBin& other) { return (mNDigitsPS < other.mNDigitsPS); }
    bool operator>(const TimeBin& other) { return (mNDigitsPS > other.mNDigitsPS); }
    bool operator<=(const TimeBin& other) { return (mNDigitsPS <= other.mNDigitsPS); }
    bool operator>=(const TimeBin& other) { return (mNDigitsPS >= other.mNDigitsPS); }
  };

  // peak search parameters
  uint32_t mTimeClusterSize;      ///< maximum size of one time cluster, in bunch crossings
  uint32_t mPeakSearchWindow;     ///< width of the peak search window, in BC units
  uint32_t mPeakSearchNbins;      ///< number of time bins for the peak search algorithm (must be an odd number >= 3)
  uint32_t mPeakSearchNDigitsMin; ///< minimum number of digits for peak candidates
  uint32_t mBinWidth;             ///< width of one time bin in the peak search algorithm, in bunch crossings
  uint32_t mNbinsInOneTF;         ///< maximum number of peak search bins in one time frame
  DigitFilter mIsGoodDigit;       ///< function to select only digits that are likely signal
  bool mImprovePeakSearch;        ///< whether to only use signal-like digits in the peak search
  bool mMergeROFs;                ///< whether to merge consecutive ROFs

  /// fill the time histogram for the peak search algorithm
  void initTimeBins();
  /// search for the next peak in the time histogram
  int32_t getNextPeak();
  /// search for the end of a given peak
  int32_t getPeakEnd(int32_t peak, int32_t max);
  /// create an output ROF containing all the digits in the [firstBin,lastBin] range of the time histogram
  void storeROF(int32_t firstBin, int32_t lastBin, int32_t peak);

  std::vector<TimeBin> mTimeBins; ///< time histogram for the peak search algorithm
  int32_t mLastSavedTimeBin;      ///< index of the last bin that has been stored in the output ROFs
  int32_t mLastSavedInputRof;     ///< index of the last bin that has been stored in the output ROFs
  int32_t mLastPeak;              ///< index of the last found peak
  int32_t mLastPeakEnd;           ///< index of the end bin of last found peak, used in the merging step

  gsl::span<const o2::mch::ROFRecord> mInputROFs; ///< input digit ROFs
  gsl::span<const o2::mch::Digit> mDigits;        ///< input digits
  ROFVector mOutputROFs{};                        ///< output time cluster ROFs
  std::vector<bool> mRofHasPeak;
  bool mDebug{false};                             ///< be more verbose
};

} // namespace mch
} // namespace o2

#endif // O2_MCH_ROFTIMECLUSTERFINDERV2_H_
