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
#include "CCDB/CCDBTimeStampUtils.h"
#include <map>
#include "TFile.h"
#include <iostream>
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[], bpo::variables_map& vm)
{
  options.add_options()(
    "host", bpo::value<std::string>()->default_value("alice-ccdb.cern.ch"), "CCDB server")(
    "path,p", bpo::value<std::vector<std::string>>()->multitoken(), "CCDB path (identifies the object) [or space separated list of paths for batch processing]")(
    "dest,d", bpo::value<std::string>()->default_value("./"), "destination path")(
    "no-preserve-path", "Do not preserve path structure. If not set, the full path structure -- reflecting the '--path' argument will be put.")(
    "outfile,o", bpo::value<std::string>()->default_value("snapshot.root"), "Name of output file. If set to \"\", the name will be determined from the uploaded content. (Will be the same in case of batch downloading multiple paths.)")(
    "timestamp,t", bpo::value<long>()->default_value(-1), "timestamp in ms - default -1 = now")(
    "created-not-before", bpo::value<long>()->default_value(0), "CCDB created-not-before time (Time Machine)")(
    "created-not-after", bpo::value<long>()->default_value(3385078236000), "CCDB created-not-after time (Time Machine)")(
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

// a simple tool to download the CCDB blob and store in a ROOT file
int main(int argc, char* argv[])
{
  bpo::options_description options("Allowed options");
  bpo::variables_map vm;
  if (!initOptionsAndParse(options, argc, argv, vm)) {
    return 1;
  }

  o2::ccdb::CcdbApi api;
  auto host = vm["host"].as<std::string>();
  api.init(host);

  std::map<std::string, std::string> filter;
  long timestamp = vm["timestamp"].as<long>();
  if (timestamp == -1) {
    timestamp = o2::ccdb::getCurrentTimestamp();
  }
  auto paths = vm["path"].as<std::vector<std::string>>();
  auto dest = vm["dest"].as<std::string>();
  if (paths.size() == 0) {
    std::cerr << "No path given";
    return 1;
  }

  std::cout << "Querying host " << host << " for path(s) " << paths[0] << " ... and timestamp " << timestamp << "\n";
  bool no_preserve_path = vm.count("no-preserve-path") == 0;
  auto filename = vm["outfile"].as<std::string>();
  auto notBefore = vm["created-not-before"].as<long>();
  auto notAfter = vm["created-not-after"].as<long>();

  bool success = true;
  for (auto& p : paths) {
    // could even multi-thread this
    success |= api.retrieveBlob(p, dest, filter, timestamp, no_preserve_path, filename, std::to_string(notAfter), std::to_string(notBefore));
  }
  return success ? 0 : 1;
}
