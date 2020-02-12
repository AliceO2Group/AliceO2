// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBQuery.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include <map>
#include "TFile.h"
#include <iostream>

// a simple tool to download the CCDB blob and store in a ROOT file
int main(int argc, char* argv[])
{

  // this is very prototypic; needs proper command line options and configurability

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " CCDBPATH TARGETPATH\n";
  }
  o2::ccdb::CcdbApi api;
  api.init("ccdb-test.cern.ch:8080");

  std::map<std::string, std::string> filter;
  api.retrieveBlob(argv[1], argv[2], filter, o2::ccdb::getCurrentTimestamp());

  return 0;
}
