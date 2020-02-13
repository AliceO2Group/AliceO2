// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <CCDB/CCDBQuery.h>
#include <iostream>

using namespace o2::ccdb;

void CCDBQuery::print() const
{
  std::cout << "path : " << path << "\n";
  std::cout << "timestamp : " << timestamp << "\n";
  std::cout << "metafilter\n{ \n";
  for (auto keyvalue : metafilter) {
    std::cout << keyvalue.first << " : " << keyvalue.second << "\n";
  }
  std::cout << "}\n";
}
