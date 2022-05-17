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

/// \file   MID/Calibration/exe/bad-channels-ccdb.cxx
/// \brief  Retrieve or upload MID bad channels
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   16 May 2022

#include <string>
#include <map>
#include <vector>
#include <boost/program_options.hpp>
#include "CCDB/CcdbApi.h"
#include "DataFormatsMID/ColumnData.h"

namespace po = boost::program_options;

const std::string BadChannelCCDBPath = "MID/Calib/BadChannels";

void queryBadChannels(const std::string ccdbUrl, long timestamp, bool verbose)
{
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> metadata;
  auto* badChannels = api.retrieveFromTFileAny<std::vector<o2::mid::ColumnData>>(BadChannelCCDBPath.c_str(), metadata, timestamp);
  std::cout << "number of bad channels = " << badChannels->size() << std::endl;
  if (verbose) {
    for (const auto& badChannel : *badChannels) {
      std::cout << badChannel << "\n";
    }
  }
}

void uploadBadChannels(const std::string ccdbUrl, long timestamp)
{
  std::vector<o2::mid::ColumnData> badChannels;

  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> md;
  std::cout << "storing default MID bad channels (valid from " << timestamp << ") to " << BadChannelCCDBPath << "\n";

  api.storeAsTFileAny(&badChannels, BadChannelCCDBPath, md, timestamp, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
}

int main(int argc, char** argv)
{
  po::variables_map vm;
  po::options_description usage("Usage");

  std::string ccdbUrl;
  long timestamp;
  bool put;
  bool query;
  bool verbose;

  uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  // clang-format off
  usage.add_options()
      ("help,h", "produce help message")
      ("ccdb,c",po::value<std::string>(&ccdbUrl)->default_value("http://ccdb-test.cern.ch:8080"),"ccdb url")
      ("timestamp,t",po::value<long>(&timestamp)->default_value(now),"timestamp for query or put")
      ("put,p",po::bool_switch(&put),"upload bad channel default object")
      ("query,q",po::bool_switch(&query),"dump bad channel object from CCDB")
      ("verbose,v",po::bool_switch(&verbose),"verbose output")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(usage);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << "Dump or upload MID bad channels CCDB object\n";
    std::cout << usage << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  if (query) {
    queryBadChannels(ccdbUrl, timestamp, verbose);
  }

  if (put) {
    uploadBadChannels(ccdbUrl, timestamp);
  }

  return 0;
}
