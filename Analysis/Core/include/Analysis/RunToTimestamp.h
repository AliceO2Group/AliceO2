// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
// Class for conversion between run number and timestamp
//
// Author: Nicolo' Jacazio on 2020-06-22

#ifndef RunToTimestamp_H
#define RunToTimestamp_H

#include <map>
#include <FairLogger.h>
#include "TNamed.h"

class RunToTimestamp : public TNamed
{
 public:
  RunToTimestamp() = default;
  ~RunToTimestamp() final = default;

  /// Checks if the converter has a particular run
  bool Has(uint runNumber) const { return mMap.count(runNumber); }

  /// Inserts a new run with a timestamp in the converter database
  bool insert(uint runNumber, long timestamp);

  /// Updates an already present run number with a new timestamp
  bool update(uint runNumber, long timestamp);

  /// Gets the timestamp of a run
  long getTimestamp(uint runNumber) const;

  /// Prints the content of the converter
  void print() const;

 private:
  std::map<uint, long> mMap;
  ClassDef(RunToTimestamp, 1) // converter class between run number and timestamp
};

#endif
