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
#include <map>
#include "TFile.h"
#include "TKey.h"
#include <iostream>

// a simple tool to inspect/print metadata content of ROOT files containing CCDB entries
// TODO: optionally print as JSON
int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " CCDBFile.root \n";
  }
  TFile file(argv[1]);

  // query the list of objects
  auto keys = file.GetListOfKeys();
  if (keys) {
    std::cout << "--- found the following objects -----\n";
    for (int i = 0; i < keys->GetEntries(); ++i) {
      auto key = static_cast<TKey*>(keys->At(i));
      if (key) {
        std::cout << key->GetName() << " of type " << key->GetClassName() << "\n";
      }
    }
  } else {
    std::cout << "--- no objects found -----\n";
  }

  auto queryinfo = o2::ccdb::CcdbApi::retrieveQueryInfo(file);
  if (queryinfo) {
    std::cout << "---found query info -----\n";
    queryinfo->print();
  } else {
    std::cout << "--- no query information found ------\n";
  }

  auto meta = o2::ccdb::CcdbApi::retrieveMetaInfo(file);
  if (meta) {
    std::cout << "---found meta info -----\n";
    for (auto keyvalue : *meta) {
      std::cout << keyvalue.first << " : " << keyvalue.second << "\n";
    }
  } else {
    std::cout << "--- no meta information found ---\n";
  }

  return 0;
}
