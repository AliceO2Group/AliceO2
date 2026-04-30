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

/// \file SectorEdgeFluctuations.h
/// \brief Class to parse and query time-dependent TPC sector edge fluctuation intervals
/// \author Matthias Kleiner <matthias.kleiner@cern.ch>

#ifndef ALICEO2_TPC_SECTOREDGEFLUCTUATIONS_H
#define ALICEO2_TPC_SECTOREDGEFLUCTUATIONS_H

#include <map>
#include <vector>
#include <string>
#include <utility>
#include "Rtypes.h"

class TTree;

namespace o2::tpc
{

/// One time interval during which a set of TPC sectors has edge fluctuations.
/// Each sector carries an optional scaling factor (default 1.0).
struct SectorEdgeInterval {
  Long64_t startTimeMS{0};                    ///< interval start, Unix time in ms
  Long64_t endTimeMS{0};                      ///< interval end,   Unix time in ms
  std::vector<std::pair<int, float>> sectors; ///< {o2SectorId, scalingFactor}
  ClassDefNV(SectorEdgeInterval, 1);
};

/// Parses and queries TPC sector edge fluctuation intervals from a CSV text file.
///
/// Expected line format (comma-separated, whitespace around tokens is ignored):
///   runNumber, startMS, endMS, durationMS, label[, SectorID[=scale], ...]
///
/// The sector list is optional. If omitted (or all tokens fail to parse), the
/// interval is applied to all 36 sectors (A0-A17 and C0-C17) with scale 1.0.
///
/// Examples:
///   560352,1732244770094,1732244771094,1000,edge distortions,A3
///   560352,1732244771344,1732244776344,5000,edge distortions,A3=1.2,C0,C1=0.6,C2,C3
///
/// If the same sector appears in multiple overlapping intervals at a queried
/// timestamp, the scale from the interval with the latest end-time is used.
class SectorEdgeFluctuations
{
 public:
  /// Load intervals from file. Clears any previously loaded data.
  /// Throws std::runtime_error if the file cannot be opened.
  bool loadFromCSVFile(const std::string& filename);

  /// dump this object to a file
  /// \param file output file
  /// \param name name of the output object
  void dumpToFile(const char* file, const char* name = "ccdb_object", const char* brName = "SectorEdgeFluctuation");

  /// load from input file (which were written using the dumpToFile method)
  /// \param inpf input file
  /// \param name name of the object in the file
  void loadFromFile(const char* inpf, const char* name = "ccdb_object", const int iEntry = 0, const char* brName = "SectorEdgeFluctuation");

  /// set this object from input tree
  void setFromTree(TTree& tree, const int iEntry = 0, const char* brName = "SectorEdgeFluctuation");

  /// Returns all {o2SectorId, scalingFactor} pairs active for the given run
  /// at the given Unix timestamp (milliseconds). Returns empty if none are active
  /// or if the run is not known.
  std::vector<std::pair<int, float>> getSectorsAtTime(int run, Long64_t timestampMS) const;

  /// Convert a sector string such as "A3" or "C14" to the O2 integer sector
  /// index (0-35). Returns -1 on parse error.
  static int parseSectorId(const std::string& sectorStr);

  /// Total number of intervals across all runs.
  size_t size() const
  {
    size_t n = 0;
    for (const auto& [run, v] : mIntervals) {
      n += v.size();
    }
    return n;
  }

  /// number of total runs stored
  size_t getNRuns() const { return mIntervals.size(); }
  bool empty() const { return mIntervals.empty(); }

  /// get stored data
  const auto& getIntervals() const { return mIntervals; }

 private:
  /// Per-run intervals, each sorted by startTimeMS.
  std::map<int, std::vector<SectorEdgeInterval>> mIntervals;

  ClassDefNV(SectorEdgeFluctuations, 1);
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_SECTOREDGEFLUCTUATIONS_H
