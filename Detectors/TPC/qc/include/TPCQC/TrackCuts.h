// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   TrackCuts.h
/// @author Thomas Klemenz, thomas.klemenz@tum.de
///

#ifndef AliceO2_TPC_TRACKCUTS_H
#define AliceO2_TPC_TRACKCUTS_H

#include "Rtypes.h"

namespace o2
{
namespace tpc
{

class TrackTPC;

namespace qc
{

/// @brief track cut class
///
/// Can be used to apply cuts on tracks during qc.
///
/// origin: TPC
/// @author Thomas Klemenz, thomas.klemenz@tum.de

class TrackCuts
{
 public:
  TrackCuts() = default;
  TrackCuts(float PMin, float PMax, float NClusMin);

  bool goodTrack(o2::tpc::TrackTPC const& track);

  void setPMin(float PMin) { mPMin = PMin; }
  void setPMax(float PMax) { mPMax = PMax; }
  void setNClusMin(float NClusMin) { mNClusMin = NClusMin; }

 private:
  float mPMin{0};     // min momentum allowed
  float mPMax{1e10};  // max momentum allowed
  float mNClusMin{0}; // min number of clusters in track allowed

  ClassDefNV(TrackCuts, 1)
};
} // namespace qc
} // namespace tpc
} // namespace o2

#endif