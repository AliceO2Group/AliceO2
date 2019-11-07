// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ANALYSIS_FLAGREASONS
#define ALICEO2_ANALYSIS_FLAGREASONS

/// \file FlagReasons.h
/// \brief classes keeping reasons for flagging time ranges
/// \author Jens Wiechula, jens.wiechula@ikf.uni-frankfurt.de
///
/// The reasons should serve in a bit mask, so can be combined

// System includes
#include <iostream>
#include <vector>
#include <string>

// ROOT includes
#include "Rtypes.h"

namespace o2
{
namespace analysis
{

/// \class FlagReasons
/// A Class for keeping several time ranges with mask of type TimeRangeFlags
class FlagReasons
{
 public:
  // singleton implementation
  static FlagReasons& instance()
  {
    static FlagReasons reasons;
    return reasons;
  }

  /// add a reason to the collection
  void addReason(std::string_view reason) { mReasonCollection.emplace_back(reason); }

  /// get the reason collection
  const auto& getReasonCollection() const { return mReasonCollection; }

  /// write data to ostream
  void streamTo(std::ostream& output) const;

  /// overloading output stream operator
  friend std::ostream& operator<<(std::ostream& output, const FlagReasons& data)
  {
    data.streamTo(output);
    return output;
  }

  /// print function
  virtual void print(/*Option_t* option = ""*/) const { std::cout << *this; }

 private:
  // default ctor, hidden for singleton
  FlagReasons() = default;

  std::vector<std::string> mReasonCollection{};
};

} // namespace analysis
} // namespace o2
#endif
