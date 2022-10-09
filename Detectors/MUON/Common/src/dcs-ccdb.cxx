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
#include "MCHConditions/DCSAliases.h"
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
#include <TFile.h>

namespace po = boost::program_options;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPMAP = std::unordered_map<DPID, std::vector<DPVAL>>;

float sum(float s, o2::dcs::DataPointValue v)
{
  union Converter {
    uint64_t raw_data;
    double value;
  } converter;
  converter.raw_data = v.payload_pt1;
  return s + converter.value;
};

std::string CcdbDpConfName()
{
  return fmt::format("{}/Config/DCSDPconfig", o2::muon::subsysname());
}

int verboseLevel;

/*
 * Return the data point values with min and max timestamps
 */
std::pair<DPVAL, DPVAL> computeTimeRange(const std::vector<DPVAL>& dps)
{
  DPVAL dmin, dmax;
  uint64_t minTime{std::numeric_limits<uint64_t>::max()};
  uint64_t maxTime{0};

  for (auto d : dps) {
    const auto ts = d.get_epoch_time();
    if (ts < minTime) {
      dmin = d;
      minTime = ts;
    }
    if (ts > maxTime) {
      dmax = d;
      maxTime = ts;
    }
  }
  return std::make_pair(dmin, dmax);
}

void dump(const std::string what, DPMAP m, int verbose)
{
  std::cout << "size of " << what << " map = " << m.size() << std::endl;
  if (verbose > 0) {
    for (auto& i : m) {
      auto v = i.second;
      auto timeRange = computeTimeRange(v);
      auto mean = std::accumulate(v.begin(), v.end(), 0.0, sum);
      if (v.size()) {
        mean /= v.size();
      }
      auto vv = v;
      auto last = std::unique(vv.begin(), vv.end());
      vv.erase(last, vv.end());

      std::cout << fmt::format("{:64s} {:4d} ({:4d} unique) values of mean {:7.2f} : ", i.first.get_alias(), v.size(), vv.size(), mean);
      if (verbose > 1) {
        for (auto dp : vv) {
          std::cout << " " << dp << "\n";
        }
      }
      std::cout << "timeRange=" << timeRange.first << " " << timeRange.second << "\n";
    }
  }
}

void doQueryHVLV(const std::string fileName)
{
  std::cout << "Reading from file " << fileName << "\n";
  std::unique_ptr<TFile> fin(TFile::Open(fileName.c_str()));
  if (fin->IsZombie()) {
    return;
  }
  TClass* cl = TClass::GetClass(typeid(DPMAP));

  DPMAP* m = static_cast<DPMAP*>(fin->GetObjectChecked("ccdb_object", cl));
  if (!m) {
    std::cerr << "Could not read ccdb_object from file " << fileName << "\n";
    return;
  }
  dump(fileName, *m, verboseLevel);
}

void doQueryHVLV(const std::string ccdbUrl, uint64_t timestamp, bool hv, bool lv)
{
  std::vector<std::string> what;
  if (hv) {
    what.emplace_back(fmt::format("{}/Calib/HV", o2::muon::subsysname()));
  }
  if (lv) {
    what.emplace_back(fmt::format("{}/Calib/LV", o2::muon::subsysname()));
  }

  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  for (auto w : what) {
    std::map<std::string, std::string> metadata;
    auto* m = api.retrieveFromTFileAny<DPMAP>(w, metadata, timestamp);
    dump(w, *m, verboseLevel);
  }
}

void doQueryDataPointConfig(const std::string ccdbUrl, uint64_t timestamp,
                            const std::string dpConfName)
{
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  using DPCONF = std::unordered_map<DPID, std::string>;
  std::map<std::string, std::string> metadata;
  auto* m = api.retrieveFromTFileAny<DPCONF>(dpConfName.c_str(), metadata, timestamp);
  std::cout << "size of dpconf map = " << m->size() << std::endl;
  if (verboseLevel > 0) {
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
    DPID::FILL(dpidtmp, a, o2::dcs::DeliveryType::DPVAL_DOUBLE);
    dpid2DataDesc[dpidtmp] = fmt::format("{}DATAPOINTS", o2::muon ::subsysname());
  }

  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> md;
  std::cout << "storing config of " << dpid2DataDesc.size()
            << o2::muon::subsysname() << " data points to "
            << CcdbDpConfName() << "\n";

  api.storeAsTFileAny(&dpid2DataDesc, CcdbDpConfName(), md, timestamp, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
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
  std::string dpConfName;
  uint64_t timestamp;
  bool lv;
  bool hv;
  bool dpconf;
  bool put;
  std::string fileName;

  uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  // clang-format off
  usage.add_options()
      ("help,h", "produce help message")
      ("ccdb,c",po::value<std::string>(&ccdbUrl)->default_value("http://localhost:6464"),"ccdb url")
      ("query,q",po::value<std::vector<std::string>>(),"what to query (if anything)")
      ("timestamp,t",po::value<uint64_t>(&timestamp)->default_value(now),"timestamp for query or put")
      ("put-datapoint-config,p",po::bool_switch(&put),"upload datapoint configuration")
      ("verbose,v",po::value<int>(&verboseLevel)->default_value(0),"verbose level")
      ("datapoint-conf-name",po::value<std::string>(&dpConfName)->default_value(CcdbDpConfName()),"dp conf name (only if not from mch or mid)")
      ("file,f",po::value<std::string>(&fileName)->default_value(""),"read from file instead of from ccdb")
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

  if (fileName.size() > 0) {
    doQueryHVLV(fileName);
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
      doQueryDataPointConfig(ccdbUrl, timestamp, dpConfName);
    }
  }

  if (put) {
    makeCCDBEntryForDCS(ccdbUrl, timestamp);
  }
  return 0;
}
