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

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBQuery.h"
#include <map>
#include "TFile.h"
#include "TKey.h"
#include <iostream>
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[], bpo::variables_map& vm)
{
  options.add_options()(
    "files,f", bpo::value<std::vector<std::string>>()->multitoken(), "Space separated list of ROOT files holding (downloaded) CCDB object")(
    "check-timestamp,t", bpo::value<long>()->default_value(-1), "Checks that validity of objects is compatible with this timestamp. In millisecond")(
    "help,h", "Produce help message.");

  try {
    bpo::store(parse_command_line(argc, argv, options), vm);
    // help
    if (vm.count("help")) {
      std::cout << options << std::endl;
      return false;
    }
    bpo::notify(vm);
  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing command line arguments; Available options:\n";

    std::cerr << options << std::endl;
    return false;
  }
  return true;
}

// A simple tool to inspect/print metadata content of ROOT files containing CCDB entries
// TODO: optionally print as JSON
int main(int argc, char* argv[])
{
  bpo::options_description options("Tool to inspect meta-data content of downloaded CCDB objects. Allowed options:");
  bpo::variables_map vm;
  if (!initOptionsAndParse(options, argc, argv, vm)) {
    return 1;
  }
  auto filenames = vm["files"].as<std::vector<std::string>>();
  auto timestamp2bchecked = vm["check-timestamp"].as<long>();
  std::vector<std::string> filesWrongValidity; // record files failing validity check

  for (auto& f : filenames) {
    std::cout << "### Loading file : " << f << "\n";
    TFile file(f.c_str());

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
      if (timestamp2bchecked > 0) {
        // retrieve Valid-From and Valid-To headers
        try {
          auto valid_from = std::stol((*meta)["Valid-From"]);
          auto valid_to = std::stol((*meta)["Valid-Until"]);
          if (!(valid_from <= timestamp2bchecked) && (timestamp2bchecked <= valid_to)) {
            std::cerr << "### ERROR: failed validity check for timestamp " << timestamp2bchecked << " not in [" << valid_from << ":" << valid_to << "]\n";
            filesWrongValidity.push_back(f);
          }
        } catch (std::exception e) {
          // no validity could be extracted;
          filesWrongValidity.push_back(f);
        }
      }
    } else {
      std::cout << "--- no meta information found ---\n";
    }
  }

  if (filesWrongValidity.size() > 0) {
    std::cerr << "### ERROR: Validity checks failed for:\n";
    for (auto& f : filesWrongValidity) {
      std::cerr << "### " << f << "\n";
    }
    return 1;
  }
  return 0;
}
