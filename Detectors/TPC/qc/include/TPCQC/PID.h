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
/// @file   PID.h
/// @author Thomas Klemenz, thomas.klemenz@tum.de
///

#ifndef AliceO2_TPC_QC_PID_H
#define AliceO2_TPC_QC_PID_H

#include <vector>
#include <map>
#include <iostream>
#include <string_view>
//root includes
#include "TH1F.h"
#include "TH1.h"
#include "TH2F.h"

//o2 includes
#include "DataFormatsTPC/Defs.h"

namespace o2
{
namespace tpc
{

class TrackTPC;

namespace qc
{

/// @brief  PID quality control class
///
/// This class is used to extract PID related variables
/// from TrackTPC objects and store it in histograms.
///
/// origin: TPC
/// @author Thomas Klemenz, thomas.klemenz@tum.de
class PID
{
 public:
  /// default constructor
  PID() = default;

  /// bool extracts intormation from track and fills it to histograms
  /// @return true if information can be extracted and filled to histograms
  bool processTrack(const o2::tpc::TrackTPC& track);

  /// Initialize all histograms
  void initializeHistograms();

  /// Reset all histograms
  void resetHistograms();

  /// Dump results to a file
  void dumpToFile(std::string filename);

  /// return a map of vectors of histograms
  std::map<std::string_view, std::vector<TH1*>> getMapOfHisto() { return mapHist2D; }
  const std::map<std::string_view, std::vector<TH1*>> getMapOfHisto() const { return mapHist2D; }
 private:
  std::map<std::string_view, std::vector<TH1*>> mapHist2D;
  ClassDefNV(PID, 1)
};
} // namespace qc
} // namespace tpc
} // namespace o2

#endif