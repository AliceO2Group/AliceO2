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
#include <boost/program_options.hpp>
#include <ctime>
#include <numeric>
#include <regex>
#include <string>
#include <vector>
#include "DataFormatsMCH/DsChannelId.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHMappingInterface/Segmentation.h"

namespace po = boost::program_options;

using BadChannelsVector = std::vector<o2::mch::DsChannelId>;

std::string ccdbPath(const std::string badChannelType)
{
  return fmt::format("MCH/Calib/{}", badChannelType);
}

void queryBadChannels(const std::string ccdbUrl,
                      const std::string badChannelType, uint64_t timestamp, bool verbose)
{
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> metadata;
  auto source = ccdbPath(badChannelType);
  auto* badChannels = api.retrieveFromTFileAny<BadChannelsVector>(source.c_str(), metadata, timestamp);
  std::cout << "number of bad channels = " << badChannels->size() << std::endl;
  if (verbose) {
    for (const auto& badChannel : *badChannels) {
      std::cout << badChannel.asString() << "\n";
    }
  }
}

void uploadBadChannels(const std::string ccdbUrl,
                       const std::string badChannelType,
                       uint64_t startTimestamp,
                       uint64_t endTimestamp,
                       std::vector<uint16_t> solarsToReject,
                       bool makeDefault)
{
  BadChannelsVector bv;

  auto det2elec = o2::mch::raw::createDet2ElecMapper<o2::mch::raw::ElectronicMapperGenerated>();

  for (auto solar : solarsToReject) {
    auto ds = o2::mch::raw::getDualSampas<o2::mch::raw::ElectronicMapperGenerated>(solar);
    for (const auto& dsDetId : ds) {
      o2::mch::mapping::Segmentation seg(dsDetId.deId());
      for (uint8_t channel = 0; channel < 64; channel++) {
        auto dsElecId = det2elec(dsDetId);
        auto padId = seg.findPadByFEE(dsDetId.dsId(), channel);
        if (seg.isValid(padId)) {
          const auto c = o2::mch::DsChannelId(solar, dsElecId->elinkId(), channel);
          bv.emplace_back(c);
        }
      }
    }
  }

  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> md;
  auto dest = ccdbPath(badChannelType);
  std::cout << "storing default MCH bad channels (valid from "
            << startTimestamp << "to " << endTimestamp << ") to "
            << dest << "\n";

  if (makeDefault) {
    md["default"] = "true";
    md["Created"] = "1";
  }
  api.storeAsTFileAny(&bv, dest, md, startTimestamp, endTimestamp);
}

int main(int argc, char** argv)
{
  po::variables_map vm;
  po::options_description usage("Usage");

  std::string ccdbUrl;
  std::string dpConfName;
  std::string badChannelType;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  bool put;
  bool query;
  bool verbose;
  bool uploadDefault;

  auto tnow = std::chrono::system_clock::now().time_since_epoch();
  using namespace std::chrono_literals;
  auto tend = tnow + 24h;

  uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(tnow).count();
  uint64_t end = std::chrono::duration_cast<std::chrono::milliseconds>(tend).count();

  // clang-format off
  usage.add_options()
      ("help,h", "produce help message")
      ("ccdb,c",po::value<std::string>(&ccdbUrl)->default_value("http://localhost:6464"),"ccdb url")
      ("starttimestamp,st",po::value<uint64_t>(&startTimestamp)->default_value(now),"timestamp for query or put - (default=now)")
      ("endtimestamp,et", po::value<uint64_t>(&endTimestamp)->default_value(end), "end of validity (for put) - default=1 day from now")
      ("put,p",po::bool_switch(&put),"upload bad channel default object")
      ("upload-default-values,u",po::bool_switch(&uploadDefault),"upload default values")
      ("type,t",po::value<std::string>(&badChannelType)->default_value("BadChannel"),"type of bad channel (BadChannel or RejectList)")
      ("query,q",po::bool_switch(&query),"dump bad channel object from CCDB")
      ("verbose,v",po::bool_switch(&verbose),"verbose output")
      ("solar,s",po::value<std::vector<uint16_t>>()->multitoken(),"solar ids to reject")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(usage);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << "This program get/set MCH bad channels CCDB object\n";
    std::cout << usage << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  if (badChannelType != "BadChannel" && badChannelType != "RejectList") {
    std::cout << "Error: badChannelType " << badChannelType << " is invalid. Only BadChannel or RejectList are legit\n";
    exit(2);
  }

  if (query) {
    queryBadChannels(ccdbUrl, badChannelType, startTimestamp, verbose);
  }

  if (put) {
    std::vector<uint16_t> solarsToReject;
    if (vm.count("solar")) {
      solarsToReject = vm["solar"].as<std::vector<uint16_t>>();
    }

    uploadBadChannels(ccdbUrl, badChannelType, startTimestamp, endTimestamp, solarsToReject, uploadDefault);
  }
  return 0;
}
