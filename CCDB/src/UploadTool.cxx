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
#include "TTree.h"
#include "TClass.h"
#include "TKey.h"
#include <iostream>
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[], bpo::variables_map& vm)
{
  options.add_options()(
    "host", bpo::value<std::string>()->default_value("ccdb-test.cern.ch:8080"), "CCDB server")(
    "path,p", bpo::value<std::string>()->required(), "CCDB path")(
    "file,f", bpo::value<std::string>()->required(), "ROOT file")(
    "key,k", bpo::value<std::string>()->required(), "Key of object to upload")(
    "meta,m", bpo::value<std::string>()->default_value(""), "List of key=value pairs for meta-information (k1=v1;k2=v2;k3=v3)")(
    "starttimestamp,st", bpo::value<long>()->default_value(-1), "timestamp - default -1 = now")(
    "endtimestamp,et", bpo::value<long>()->default_value(-1), "end of validity - default -1 = 1 day from now")(
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

// Remove leading whitespace
std::string ltrimSpace(std::string src)
{
  return src.erase(0, src.find_first_not_of(' '));
}

// Remove trailing whitespace
std::string rtrimSpace(std::string src)
{
  return src.erase(src.find_last_not_of(' ') + 1);
}

// Remove leading/trailing whitespace
std::string trimSpace(std::string const& src)
{
  return ltrimSpace(rtrimSpace(src));
}

// Split a given string on a delim character, return vector of tokens
// If trim is true, then also remove leading/trailing whitespace of each token.
std::vector<std::string> splitString(const std::string& src, char delim, bool trim = false)
{
  std::stringstream ss(src);
  std::string token;
  std::vector<std::string> tokens;

  while (std::getline(ss, token, delim)) {
    token = (trim ? trimSpace(token) : token);
    if (!token.empty()) {
      tokens.push_back(std::move(token));
    }
  }

  return tokens;
}

// a simple tool to take an abitrary object in a ROOT file and upload it to CCDB including some
// meta information
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
  long starttimestamp = vm["starttimestamp"].as<long>();
  if (starttimestamp == -1) {
    starttimestamp = o2::ccdb::getCurrentTimestamp();
  }
  long endtimestamp = vm["endtimestamp"].as<long>();
  if (endtimestamp == -1) {
    constexpr long SECONDSPERDAY = 1 * 24 * 60 * 60;
    endtimestamp = o2::ccdb::getFutureTimestamp(SECONDSPERDAY);
  }

  auto filename = vm["file"].as<std::string>();
  auto path = vm["path"].as<std::string>();
  auto keyname = vm["key"].as<std::string>();

  // Take a vector of strings with elements of form a=b, and
  // return a vector of pairs with each pair of form <a, b>
  auto toKeyValPairs = [](std::vector<std::string> const& tokens) {
    std::vector<std::pair<std::string, std::string>> pairs;

    for (auto& token : tokens) {
      auto keyval = splitString(token, '=');
      if (keyval.size() != 2) {
        // LOG(fatal) << "Illegal command-line key/value string: " << token;
        continue;
      }

      std::pair<std::string, std::string> pair = std::make_pair(keyval[0], trimSpace(keyval[1]));
      pairs.push_back(pair);
    }

    return pairs;
  };
  auto metastring = vm["meta"].as<std::string>();
  auto keyvalues = toKeyValPairs(splitString(metastring, ';', true));

  // fill meta map
  std::map<std::string, std::string> meta;
  for (auto& p : keyvalues) {
    meta[p.first] = p.second;
  }

  TFile f(filename.c_str());
  auto key = f.GetKey(keyname.c_str());
  if (key) {
    // get type of key
    auto classname = key->GetClassName();
    auto tcl = TClass::GetClass(classname);
    auto object = f.Get<void>(keyname.c_str());
    if (tcl->InheritsFrom("TTree")) {
      auto tree = static_cast<TTree*>(object);
      tree->LoadBaskets(0x1L << 32); // make tree memory based
      tree->SetDirectory(nullptr);
    }
    // convert classname to typeinfo
    // typeinfo
    auto ti = tcl->GetTypeInfo();

    std::cout << " Uploading an object of type " << key->GetClassName()
              << " to path " << path << " with timestamp validy from " << starttimestamp
              << " to " << endtimestamp << "\n";

    api.storeAsTFile_impl(object, *ti, path, meta, starttimestamp, endtimestamp);
  } else {
    std::cerr << "Key " << keyname << " does not exist\n";
  }

  return 0;
}
