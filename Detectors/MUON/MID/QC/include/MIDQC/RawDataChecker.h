// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDQC/RawDataChecker.h
/// \brief  Class to check the raw data
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   9 December 2019
#ifndef O2_MID_RAWDATACHECKER_H
#define O2_MID_RAWDATACHECKER_H

#include <array>
#include <string>
#include <gsl/gsl>
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/LocalBoardRO.h"
#include "MIDQC/GBTRawDataChecker.h"

namespace o2
{
namespace mid
{
class RawDataChecker
{
 public:
  void init(const CrateMasks& masks);
  bool process(gsl::span<const LocalBoardRO> localBoards, gsl::span<const ROFRecord> rofRecords, gsl::span<const ROFRecord> pageRecords);
  /// Gets the number of processed events
  unsigned int getNEventsProcessed() const;
  /// Gets the number of faulty events
  unsigned int getNEventsFaulty() const;
  /// Gets the number of busy raised
  unsigned int getNBusyRaised() const;
  /// Gets the debug message
  std::string getDebugMessage() const { return mDebugMsg; }
  void clear();

 private:
  std::array<GBTRawDataChecker, crateparams::sNGBTs> mCheckers{}; /// GBT raw data checker
  std::string mDebugMsg{};                                        /// Debug message
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_RAWDATACHECKER_H */
