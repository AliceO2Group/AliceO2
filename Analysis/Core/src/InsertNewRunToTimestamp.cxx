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
// A simple tool to produce objects for the conversion from run number to timestamp.
// The tool uploads the 'RunNumber -> timestamp' converter to CCDB.
// If no converter object is found in CCDB a new one is created and uploaded.
//
// Author: Nicolo' Jacazio on 2020-06-22

#include "Analysis/RunToTimestamp.h"
#include "CCDB/CcdbApi.h"
#include <boost/program_options.hpp>
#include <FairLogger.h>

namespace bpo = boost::program_options;

bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[], bpo::variables_map& vm)
{
  options.add_options()(
    "run,r", bpo::value<uint>()->required(), "Run number to use")(
    "timestamp,t", bpo::value<long>()->required(), "Timestamp to use equivalent to the run number")(
    "path,p", bpo::value<std::string>()->default_value("Test/RunToTimestamp"), "Path to the object in the CCDB repository")(
    "url,u", bpo::value<std::string>()->default_value("http://ccdb-test.cern.ch:8080"), "URL of the CCDB database")(
    "start,s", bpo::value<long>()->default_value(0), "Start timestamp of object validity")(
    "stop,S", bpo::value<long>()->default_value(4108971600000), "Stop timestamp of object validity")(
    "update,u", bpo::value<int>()->default_value(0), "Flag to update the object instead of inserting the new timestamp")(
    "verbose,v", bpo::value<int>()->default_value(0), "Verbose level 0, 1")(
    "help,h", "Produce help message.");

  try {
    bpo::store(parse_command_line(argc, argv, options), vm);

    // help
    if (vm.count("help")) {
      LOG(INFO) << options;
      return false;
    }

    bpo::notify(vm);
  } catch (const bpo::error& e) {
    LOG(ERROR) << e.what() << "\n";
    LOG(ERROR) << "Error parsing command line arguments; Available options:";
    LOG(ERROR) << options;
    return false;
  }
  return true;
}

int main(int argc, char* argv[])
{
  bpo::options_description options("Allowed options");
  bpo::variables_map vm;
  if (!initOptionsAndParse(options, argc, argv, vm)) {
    return 1;
  }

  o2::ccdb::CcdbApi api;
  const std::string url = vm["url"].as<std::string>();
  api.init(url);
  if (!api.isHostReachable()) {
    LOG(WARNING) << "CCDB host " << url << " is not reacheable, cannot go forward";
    return 1;
  }

  std::string path = vm["path"].as<std::string>();
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string>* headers;
  long start = vm["start"].as<long>();
  long stop = vm["stop"].as<long>();

  RunToTimestamp* converter = api.retrieveFromTFileAny<RunToTimestamp>(path, metadata, -1, headers);

  if (!converter) {
    LOG(INFO) << "Did not retrieve run number to timestamp converter, creating a new one!";
    converter = new RunToTimestamp();
  } else {
    LOG(INFO) << "Retrieved run number to timestamp converter from ccdb url" << url;
  }

  if (vm["update"].as<int>())
    converter->update(vm["run"].as<uint>(), vm["timestamp"].as<long>());
  else
    converter->insert(vm["run"].as<uint>(), vm["timestamp"].as<long>());

  if (vm["verbose"].as<int>())
    converter->print();

  api.storeAsTFileAny(converter, path, metadata, start, stop);

  return 0;
}
