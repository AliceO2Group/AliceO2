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

#ifndef O2_MCH_ROFTIMECLUSTERFINDER_H_
#define O2_MCH_ROFTIMECLUSTERFINDER_H_

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

class ROFTimeClusterFinder
{
 public:
  using ROFVector = std::vector<o2::mch::ROFRecord>;

  ROFTimeClusterFinder(gsl::span<const o2::mch::ROFRecord> rofs, gsl::span<const o2::mch::Digit> digits, uint32_t timeClusterSize, uint32_t nBins, bool improvePeakSearch, bool debug);
  ~ROFTimeClusterFinder() = default;

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
    int32_t mFirstIdx{-1}; ///< index of the first digit ROF in the bin
    int32_t mLastIdx{-1};  ///< index of the last digit ROF in the bin
    uint32_t mSize{0};     ///< number of digit ROFs in the bin
    uint32_t mNDigits{0};  ///< number of digits in the bin
    uint32_t mNDigitsPS{0}; ///< number of digits in the bin for the peak search step

    TimeBin() = default;

    bool empty() { return mNDigits == 0; }

    bool operator<(const TimeBin& other) { return (mNDigitsPS < other.mNDigitsPS); }
    bool operator>(const TimeBin& other) { return (mNDigitsPS > other.mNDigitsPS); }
    bool operator<=(const TimeBin& other) { return (mNDigitsPS <= other.mNDigitsPS); }
    bool operator>=(const TimeBin& other) { return (mNDigitsPS >= other.mNDigitsPS); }
  };

  // peak search parameters
  static constexpr uint32_t sMaxOrbitsInTF = 256;                              ///< upper limit of the time frame size
  static constexpr uint32_t sBcInOneOrbit = o2::constants::lhc::LHCMaxBunches; ///< number of bunch-crossings in one orbit
  static constexpr uint32_t sBcInOneTF = sBcInOneOrbit * sMaxOrbitsInTF;       ///< maximum number of bunch-crossings in one time frame

  uint32_t mTimeClusterSize;  ///< maximum size of one time cluster, in bunch crossings
  uint32_t mNbinsInOneWindow; ///< number of time bins considered for the peak search
  double mBinWidth;           ///< width of one time bin in the peak search algorithm, in bunch crossings
  uint32_t mNbinsInOneTF;     ///< maximum number of peak search bins in one time frame
  DigitFilter mIsGoodDigit;   ///< function to select only digits that are likely signal
  bool mImprovePeakSearch;    ///< whether to only use signal-like digits in the peak search

  /// fill the time histogram for the peak search algorithm
  void initTimeBins();
  /// search for the next peak in the time histogram
  int32_t getNextPeak();
  /// create an output ROF containing all the digits in the [firstBin,lastBin] range of the time histogram
  void storeROF(int32_t firstBin, int32_t lastBin);

  std::vector<TimeBin> mTimeBins; ///< time histogram for the peak search algorithm
  int32_t mLastSavedTimeBin;      ///< index of the last bin that has been stored in the output ROFs

  gsl::span<const o2::mch::ROFRecord> mInputROFs; ///< input digit ROFs
  gsl::span<const o2::mch::Digit> mDigits;        ///< input digits
  ROFVector mOutputROFs{};                        ///< output time cluster ROFs
  bool mDebug{false};                             ///< be more verbose
};

} // namespace mch
} // namespace o2

#endif // O2_MCH_ROFTIMECLUSTERFINDER_H_
