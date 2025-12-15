// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef INDEXJSONHELPERS_H
#define INDEXJSONHELPERS_H

#include <Framework/AnalysisHelpers.h>

namespace o2::framework
{
struct IndexJSONHelpers {
  static std::vector<o2::soa::IndexRecord> read(std::istream& s);
  static void write(std::ostream& o, std::vector<o2::soa::IndexRecord>& irs);
};

} // namespace o2::framework

#endif // INDEXJSONHELPERS_H
