// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Analysis/RunToTimestamp.h"
#include "CCDB/CcdbApi.h"
// #include "CCDB/CCDBQuery.h"
// #include "CCDB/CCDBTimeStampUtils.h"
// #include <map>
// #include "TFile.h"
// #include "TClass.h"
// #include "TKey.h"
// #include <iostream>
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
    "stop,S", bpo::value<long>()->default_value(1592870400000), "Stop timestamp of object validity")(
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

// a simple tool to produce objects for the conversion from run number to timestamp
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

  converter->insert(vm["run"].as<uint>(), vm["timestamp"].as<long>());

  if (vm["verbose"].as<int>())
    converter->print();

  api.storeAsTFileAny(converter, path, metadata, start, stop);

  //   std::map<std::string, std::string> filter;
  //   long starttimestamp = vm["starttimestamp"].as<long>();
  //   if (starttimestamp == -1) {
  //     starttimestamp = o2::ccdb::getCurrentTimestamp();
  //   }
  //   long endtimestamp = vm["endtimestamp"].as<long>();
  //   if (endtimestamp == -1) {
  //     constexpr long SECONDSPERYEAR = 365 * 24 * 60 * 60;
  //     endtimestamp = o2::ccdb::getFutureTimestamp(SECONDSPERYEAR);
  //   }

  //   auto filename = vm["file"].as<std::string>();
  //   auto path = vm["path"].as<std::string>();
  //   auto keyname = vm["key"].as<std::string>();

  //   // Take a vector of strings with elements of form a=b, and
  //   // return a vector of pairs with each pair of form <a, b>
  //   auto toKeyValPairs = [](std::vector<std::string> const& tokens) {
  //     std::vector<std::pair<std::string, std::string>> pairs;

  //     for (auto& token : tokens) {
  //       auto keyval = splitString(token, '=');
  //       if (keyval.size() != 2) {
  //         // LOG(FATAL) << "Illegal command-line key/value string: " << token;
  //         continue;
  //       }

  //       std::pair<std::string, std::string> pair = std::make_pair(keyval[0], trimSpace(keyval[1]));
  //       pairs.push_back(pair);
  //     }

  //     return pairs;
  //   };
  //   auto metastring = vm["meta"].as<std::string>();
  //   auto keyvalues = toKeyValPairs(splitString(metastring, ';', true));

  //   // fill meta map
  //   std::map<std::string, std::string> meta;
  //   for (auto& p : keyvalues) {
  //     meta[p.first] = p.second;
  //   }

  //   TFile f(filename.c_str());
  //   auto key = f.GetKey(keyname.c_str());
  //   if (key) {
  //     // get type of key
  //     auto classname = key->GetClassName();
  //     auto object = f.Get<void>(keyname.c_str());
  //     // convert classname to typeinfo
  //     auto tcl = TClass::GetClass(classname);
  //     // typeinfo
  //     auto ti = tcl->GetTypeInfo();

  //     std::cout << " Uploading an object of type " << key->GetClassName()
  //               << " to path " << path << " with timestamp validy from " << starttimestamp
  //               << " to " << endtimestamp << "\n";

  //     api.storeAsTFile_impl(object, *ti, path, meta, starttimestamp, endtimestamp);
  //   } else {
  //     std::cerr << "Key " << keyname << " does not exist\n";
  //   }

  return 0;
}
