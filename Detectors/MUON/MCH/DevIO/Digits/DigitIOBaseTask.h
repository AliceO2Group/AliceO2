// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#pragma once

#include "Framework/ConfigParamSpec.h"
#include <cstdint>
#include <gsl/span>
#include <numeric>
#include <vector>

namespace o2::framework
{
class InitContext;
} // namespace o2::framework

namespace o2::mch
{
class Digit;
class ROFRecord;
}; // namespace o2::mch

namespace o2::mch::io
{

/** 
 * DigitIOBaseTask implements the commonalities between reader and writer
 * tasks, like the handling of the common options.
 */

class DigitIOBaseTask
{
 protected:
  size_t mMaxNofTimeFrames{std::numeric_limits<size_t>::max()}; // max number of timeframes to process
  size_t mNofProcessedTFs{0};                                   // actual number of timeframes processed so far
  size_t mFirstTF{0};                                           // first timeframe to process
  size_t mTFid{0};                                              // current timeframe index
  bool mPrintDigits = false;                                    // print digits
  bool mPrintTFs = false;                                       // print number of rofs and digits per tf

 public:
  /**
   * Init data members from options
   */
  void init(o2::framework::InitContext& ic);

  /**
   * Make a full screen dump of the digits and rofs arrays.
   */
  void printFull(gsl::span<const Digit> digits,
                 gsl::span<const ROFRecord> rofs) const;

  /**
   * Make a brief screen dump of the digits and rofs arrays (just showing
   * number of items in each)
   */
  void printSummary(gsl::span<const Digit> digits,
                    gsl::span<const ROFRecord> rofs,
                    const char* suffix = "") const;

  /**
   * Decide, depending on the current TFid being processed, if it should
   * be processed or not.
   */
  bool shouldProcess() const;

  /**
   * Increment the number of timeframes that have been processed so far.
   */
  void incNofProcessedTFs();

  /**
   * Increment the timeframe id (last one that has been processed).
   */
  void incTFid();
};

/**
 * Define commonly used options, like max number of tfs, first tf, etc...
 */
std::vector<o2::framework::ConfigParamSpec> getCommonOptions();

} // namespace o2::mch::io
