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
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[], bpo::variables_map& vm)
{
  options.add_options()(
    "host", bpo::value<std::string>()->default_value("ccdb-test.cern.ch:8080"), "CCDB server")(
    "path,p", bpo::value<std::string>(), "CCDB path")(
    "dest,d", bpo::value<std::string>(), "destination path")(
    "timestamp,t", bpo::value<long>()->default_value(-1), "timestamp - default -1 = now")(
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
    return -1;
  }

  o2::ccdb::CcdbApi api;
  auto host = vm["host"].as<std::string>();
  api.init(host);

  std::map<std::string, std::string> filter;
  long timestamp = vm["timestamp"].as<long>();
  if (timestamp == -1) {
    timestamp = o2::ccdb::getCurrentTimestamp();
  }
  auto path = vm["path"].as<std::string>();
  auto dest = vm["dest"].as<std::string>();

  std::cout << "Querying host " << host << " for path " << path << " and timestamp " << timestamp << "\n";
  api.retrieveBlob(path, dest, filter, timestamp);

  return 0;
}
