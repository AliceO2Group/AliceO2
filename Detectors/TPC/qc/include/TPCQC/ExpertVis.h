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
/// @file   ExpertVis.h
/// @author Berkin Ulukutlu, berkin.ulukutlu@cern.ch
///

#ifndef AliceO2_TPC_QC_ExpertVis_H
#define AliceO2_TPC_QC_ExpertVis_H

#include <memory>

// root includes
#include "THnSparse.h"

// o2 includes

namespace o2::tpc
{

class TrackTPC;

namespace qc
{

/// @brief  Exprt visualization quality control class
///
/// This class is used to create THnSparse files
/// from TrackTPC objects to be used in offline visualization
///
/// origin: TPC
/// @author Berkin Ulukutlu, berkin.ulukutlu@cern.ch
class ExpertVis
{
 public:
  /// default constructor
  ExpertVis() = default;

  /// bool extracts intormation from track and fills it to THnSparse
  /// @return true if information can be extracted and filled to THnSparse
  bool processTrack(const o2::tpc::TrackTPC& track);

  /// Initialize all histograms
  void initializeHistograms();

  /// Reset all histograms
  void resetHistograms();

  /// Dump results to a file
  void dumpToFile(std::string_view filename);

  /// get ND histogram
  THnSparse* getHistogramPID() { return mPIDND.get(); }
  const THnSparse* getHistogramPID() const { return mPIDND.get(); }

  THnSparse* getHistogramTracks() { return mTracksND.get(); }
  const THnSparse* getHistogramTracks() const { return mTracksND.get(); }

 private:
  std::unique_ptr<THnSparse> mPIDND;
  std::unique_ptr<THnSparse> mTracksND;

  ClassDefNV(ExpertVis, 1)
};
} // namespace qc
} // namespace o2::tpc

#endif