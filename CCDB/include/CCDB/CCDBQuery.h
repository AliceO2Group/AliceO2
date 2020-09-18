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
/// \file   CCDBQuery.h
/// \author Barthelemy von Haller, Sandro Wenzel
///

#ifndef ALICEO2_CCDBQUERY_H
#define ALICEO2_CCDBQUERY_H

#include <string>
#include <map>
#include "Rtypes.h" // for ClassDefNV

class TFile;

namespace o2
{
namespace ccdb
{

// encapsulate information how CCDB was queried
struct CCDBQuery {
  CCDBQuery() = default;
  CCDBQuery(std::string const& p, std::map<std::string, std::string> const& f, long ts) : path{p}, metafilter{f}, timestamp{ts} {}

  std::string path;                              // the path of the object to fetch
  std::map<std::string, std::string> metafilter; // filter on metadata
  long timestamp = -1;                           // the timestamp of the query

  void print() const;

  ClassDefNV(CCDBQuery, 1);
};

} // namespace ccdb
} // namespace o2

#endif
