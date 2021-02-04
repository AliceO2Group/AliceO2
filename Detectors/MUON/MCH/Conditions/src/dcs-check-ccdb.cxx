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
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include <boost/program_options.hpp>
#include <ctime>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace po = boost::program_options;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPMAP = std::unordered_map<DPID, std::vector<DPVAL>>;

int main(int argc, char** argv)
{
  po::variables_map vm;
  po::options_description usage("Usage");

  std::string ccdbUrl;
  uint64_t timestamp;
  bool lv;
  bool hv;
  bool verbose;

  std::time_t now = std::time(nullptr);

  // clang-format off
  usage.add_options()
      ("help,h", "produce help message")
      ("ccdb",po::value<std::string>(&ccdbUrl)->default_value("http://localhost:6464"),"ccdb url")
      ("timestamp,t",po::value<uint64_t>(&timestamp)->default_value(now),"timestamp to query")
      ("hv",po::bool_switch(&hv),"query HV")
      ("lv",po::bool_switch(&lv),"query LV")
      ("verbose,v",po::bool_switch(&verbose),"verbose output")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(usage);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << "This program printout summary information from MCH DCS entries.\n";
    std::cout << usage << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  if (!hv && !lv) {
    std::cout << "Must specify at least one of --hv or --lv\n";
    std::cout << usage << "\n";
    return 3;
  }

  std::vector<std::string> what;
  if (hv) {
    what.emplace_back("MCH/HV");
  }
  if (lv) {
    what.emplace_back("MCH/LV");
  }

  // union Converter {
  //   uint64_t raw_data;
  //   T t_value;
  // };
  // if (dpcom.id.get_type() != dt) {
  //   throw std::runtime_error("DPCOM is of unexpected type " + o2::dcs::show(dt));
  // }
  // Converter converter;
  // converter.raw_data = dpcom.data.payload_pt1;

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
  return 0;
}
