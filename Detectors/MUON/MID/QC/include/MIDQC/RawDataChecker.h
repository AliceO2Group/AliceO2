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
#include "MIDRaw/ElectronicsDelay.h"
#include "DataFormatsMID/ROBoard.h"
#include "MIDQC/GBTRawDataChecker.h"

namespace o2
{
namespace mid
{
class RawDataChecker
{
 public:
  void init(const CrateMasks& masks);
  bool process(gsl::span<const ROBoard> localBoards, gsl::span<const ROFRecord> rofRecords, gsl::span<const ROFRecord> pageRecords);
  bool checkMissingLinks(bool clear = true);
  /// Gets the number of processed events
  unsigned int getNEventsProcessed() const;
  /// Gets the number of faulty events
  unsigned int getNEventsFaulty() const;
  /// Gets the number of busy raised
  unsigned int getNBusyRaised() const;
  /// Gets the debug message
  std::string getDebugMessage() const { return mDebugMsg; }
  void clear(bool all = false);

  /// Sets the delay in the electronics
  void setElectronicsDelay(const ElectronicsDelay& electronicsDelay) { mElectronicsDelay = electronicsDelay; }

  void setSyncTrigger(uint32_t syncTrigger);

 private:
  std::array<GBTRawDataChecker, crateparams::sNGBTs> mCheckers{}; /// GBT raw data checker
  std::string mDebugMsg{};                                        /// Debug message
  ElectronicsDelay mElectronicsDelay{};                           /// Delays in the electronics
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_RAWDATACHECKER_H */
