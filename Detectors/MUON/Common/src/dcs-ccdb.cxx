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
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#if defined(MUON_SUBSYSTEM_MCH)
#include "MCHConditions/DCSNamer.h"
#elif defined(MUON_SUBSYSTEM_MID)
#include "MIDConditions/DCSNamer.h"
#endif
#include <boost/program_options.hpp>
#include <ctime>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#include "subsysname.h"

namespace po = boost::program_options;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPMAP = std::unordered_map<DPID, std::vector<DPVAL>>;

std::string CcdbDpConfName()
{
  return fmt::format("{}/DCSconfig", o2::muon::subsysname());
}

bool verbose;

void doQueryHVLV(const std::string ccdbUrl, uint64_t timestamp, bool hv, bool lv)
{
  std::vector<std::string> what;
  if (hv) {
    what.emplace_back(fmt::format("{}/HV", o2::muon::subsysname()));
  }
  if (lv) {
    what.emplace_back(fmt::format("{}/LV", o2::muon::subsysname()));
  }

  auto sum =
    [](float s, o2::dcs::DataPointValue v) {
      union Converter {
        uint64_t raw_data;
        double value;
      } converter;
      converter.raw_data = v.payload_pt1;
      return s + converter.value;
    };

  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  for (auto w : what) {
    std::map<std::string, std::string> metadata;
    auto* m = api.retrieveFromTFileAny<DPMAP>(w, metadata, timestamp);
    std::cout << "size of " << w << " map = " << m->size() << std::endl;
    if (verbose) {
      for (auto& i : *m) {
        auto v = i.second;
        auto mean = std::accumulate(v.begin(), v.end(), 0.0, sum);
        if (v.size()) {
          mean /= v.size();
        }
        std::cout << fmt::format("{:64s} {:4d} values of mean {:7.2f}\n", i.first.get_alias(), v.size(),
                                 mean);
      }
    }
  }
}

void doQueryDataPointConfig(const std::string ccdbUrl, uint64_t timestamp)
{
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  using DPCONF = std::unordered_map<DPID, std::string>;
  std::map<std::string, std::string> metadata;
  auto* m = api.retrieveFromTFileAny<DPCONF>(CcdbDpConfName().c_str(), metadata, timestamp);
  std::cout << "size of dpconf map = " << m->size() << std::endl;
  if (verbose) {
    for (auto& i : *m) {
      std::cout << i.second << " " << i.first << "\n";
    }
  }
}

void makeCCDBEntryForDCS(const std::string ccdbUrl, uint64_t timestamp)
{
  std::unordered_map<DPID, std::string> dpid2DataDesc;
#if defined(MUON_SUBSYSTEM_MCH)
  auto aliases = o2::mch::dcs::aliases();
#elif defined(MUON_SUBSYSTEM_MID)
  auto aliases = o2::mid::dcs::aliases();
#endif

  DPID dpidtmp;
  for (const auto& a : aliases) {
    DPID::FILL(dpidtmp, a, o2::dcs::DeliveryType::RAW_DOUBLE);
    dpid2DataDesc[dpidtmp] = fmt::format("{}DATAPOINTS", o2::muon ::subsysname());
  }

  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> md;
  std::cout << "storing config of " << dpid2DataDesc.size()
            << o2::muon::subsysname() << " data points to "
            << CcdbDpConfName() << "\n";

  api.storeAsTFileAny(&dpid2DataDesc, CcdbDpConfName(), md, timestamp);
}

bool match(const std::vector<std::string>& queries, const char* pattern)
{
  return std::find_if(queries.begin(), queries.end(), [pattern](std::string s) { return std::regex_match(s, std::regex(pattern, std::regex::extended | std::regex::icase)); }) != queries.end();
}

int main(int argc, char** argv)
{
  po::variables_map vm;
  po::options_description usage("Usage");

  std::string ccdbUrl;
  uint64_t timestamp;
  bool lv;
  bool hv;
  bool dpconf;
  bool put;

  std::time_t now = std::time(nullptr);

  // clang-format off
  usage.add_options()
      ("help,h", "produce help message")
      ("ccdb,c",po::value<std::string>(&ccdbUrl)->default_value("http://localhost:6464"),"ccdb url")
      ("query,q",po::value<std::vector<std::string>>(),"what to query (if anything)")
      ("timestamp,t",po::value<uint64_t>(&timestamp)->default_value(now),"timestamp for query or put")
      ("put-datapoint-config,p",po::bool_switch(&put),"upload datapoint configuration")
      ("verbose,v",po::bool_switch(&verbose),"verbose output")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(usage);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << "This program printout summary information from "
              << o2::muon::subsysname() << " DCS entries.\n";
    std::cout << usage << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  if (vm.count("query")) {
    auto query = vm["query"].as<std::vector<std::string>>();

    hv = match(query, ".*(hv)");
#if defined(MUON_SUBSYSTEM_MCH)
    lv = match(query, ".*(lv)");
#else
    lv = false;
#endif
    dpconf = match(query, ".*(dpconf)");

    if (!hv && !lv && !dpconf) {
      std::cout << "Must specify at least one of dpconf,hv";
#if defined(MUON_SUBSYSTEM_MCH)
      std::cout << ",lv";
#endif
      std::cout << " parameter to --query option\n";
      std::cout
        << usage << "\n";
      return 3;
    }

    if (hv || lv) {
      doQueryHVLV(ccdbUrl, timestamp, hv, lv);
    }

    if (dpconf) {
      doQueryDataPointConfig(ccdbUrl, timestamp);
    }
  }

  if (put) {
    makeCCDBEntryForDCS(ccdbUrl, timestamp);
  }
  return 0;
}
