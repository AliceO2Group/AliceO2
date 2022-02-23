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

#include <boost/program_options.hpp>
#include <ctime>
#include <chrono>
#include "DetectorsBase/GeometryManager.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CommonUtils/NameConf.h"
#include <TGeoManager.h>

using CcdbApi = o2::ccdb::CcdbApi;
namespace bpo = boost::program_options;

void createAlignedGeometry(long timeToPick, std::string ccdbServer = "")
{
  if (timeToPick == 0) {
    timeToPick = o2::ccdb::getCurrentTimestamp();
  }
  auto& cm = o2::ccdb::BasicCCDBManager::instance();
  if (!ccdbServer.empty()) {
    cm.setURL(ccdbServer);
  }
  cm.setTimestamp(timeToPick);
  cm.get<TGeoManager>("GLO/Config/Geometry");
  if (!o2::base::GeometryManager::isGeometryLoaded()) {
    throw std::runtime_error(fmt::format("Failed to load geometry from {} for timestamp {}", cm.getURL(), timeToPick));
  }
  gGeoManager->SetName(std::string(o2::base::NameConf::CCDBOBJECT).c_str());
  o2::base::GeometryManager::applyMisalignent();
  auto fnm = o2::base::NameConf::getAlignedGeomFileName();
  gGeoManager->Export(fnm.c_str());
  LOG(info) << "Stored to local file " << fnm;
}

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general(
    "Create aligned geometry from CCDB ideal geometry and alignment entries\n"
    "Usage:\n  " +
    std::string(argv[0]) +
    "");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("timestamp,t", bpo::value<long>()->default_value(0), "timestamp to use for CCDB query (0 = now)");
    add_option("ccdb-server", bpo::value<std::string>()->default_value("http://alice-ccdb.cern.ch"), "CCDB server to query");
    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);
    if (vm.count("help")) {
      std::cout << opt_general << std::endl;
      exit(0);
    }
    bpo::notify(vm);
  } catch (bpo::error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl
              << std::endl;
    std::cerr << opt_general << std::endl;
    exit(1);
  } catch (std::exception& e) {
    std::cerr << e.what() << ", application will now exit" << std::endl;
    exit(2);
  }

  createAlignedGeometry(
    vm["timestamp"].as<long>(),
    vm["ccdb-server"].as<std::string>());
}
