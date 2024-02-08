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

// example to run:
//
#include <boost/program_options.hpp>
#include <filesystem>
#include <TFile.h>
#include <TStopwatch.h>
#include "CommonUtils/StringUtils.h"
#include <CCDB/BasicCCDBManager.h>
#include <iostream>
#include <vector>
#include <string>
namespace bpo = boost::program_options;
//
// get object from ccdb  auto pp = ccdbMgr.getSpecific<std::vector<long>>("CTP/Calib/OrbitResetTest")
int main(int argc, char** argv)
{
  const std::string testCCDB = "http://ccdb-test.cern.ch:8080";
  const std::string prodCCDB = "http://o2-ccdb.internal";
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       " Write orbit staff to ccdb\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;
  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("output-file,o", bpo::value<std::string>()->default_value("none"), "output file name, none - file not created");
    add_option("output-dir,d", bpo::value<std::string>()->default_value("./"), "output dir");
    add_option("ccdb", bpo::value<std::string>()->default_value("test"), "choose databse: test- test ccdb; prod - production ccdb; else ccdb parameter");
    add_option("action,a", bpo::value<std::string>()->default_value(""), "sox - first orbit otherwise orbit reset");
    add_option("run-number,r", bpo::value<int64_t>()->default_value(123), "run number");
    add_option("testReset,t", bpo::value<bool>()->default_value(0), "0 = CTP/Calib/OrbitReset; 1 = CTP/Calib/OrbitResetTest");
    add_option("sox-orbit,x", bpo::value<int64_t>()->default_value(0), "SOX orbit");

    //
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
  std::string action = vm["action"].as<std::string>();
  std::vector<int64_t> vect;
  std::string ccdbPath;
  auto now = std::chrono::system_clock::now();
  long tt = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  if (action == "sox") {
    // write to CTP/Calib/FirstRunOrbit
    std::cout << "===> FirsRunOrbit" << std::endl;
    vect.push_back(tt);
    vect.push_back(vm["run-number"].as<int64_t>());
    vect.push_back(vm["sox-orbit"].as<int64_t>());
    ccdbPath = "CTP/Calib/FirstRunOrbit";
  } else {
    // write to CTP/Calib/OrbitReset
    std::cout << "===> ResetOrbit" << std::endl;
    vect.push_back(tt);
    ccdbPath = "CTP/Calib/OrbitReset";
    if (vm["testReset"].as<bool>()) {
      ccdbPath += "Test";
    }
  }
  //
  std::string ccdbAddress;
  if (vm["ccdb"].as<std::string>() == "prod") {
    ccdbAddress = prodCCDB;
  } else if (vm["ccdb"].as<std::string>() == "test") {
    ccdbAddress = testCCDB;
  } else {
    ccdbAddress = vm["ccdb"].as<std::string>();
  }
  std::cout << " Writing to db:" << ccdbAddress << std::endl;
  if (ccdbAddress != "none") {
    // auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
    o2::ccdb::CcdbApi api;
    api.init(ccdbAddress.c_str());
    std::map<std::string, std::string> metadata;
    long tmin = tt;
    long tmax = tmin + 381928219;
    if (action == "sox") {
      int64_t runnum = vm["run-number"].as<int64_t>();
      metadata["runNumber"] = std::to_string(runnum);
      std::cout << "Storing:" << ccdbPath << " " << metadata["runNumber"] << " tmin:" << tmin << " tmax:" << tmax << std::endl;
      api.storeAsTFileAny(&(vect), ccdbPath, metadata, tmin, tmax);
    } else {
      std::cout << "Storing:" << ccdbPath << " tmin:" << tmin << " tmax:" << tmax << std::endl;
      api.storeAsTFileAny(&(vect), ccdbPath, metadata, tmin, tmax);
    }
  }
  //
  if (vm["output-file"].as<std::string>() != "none") {
    std::string file = vm["output-dir"].as<std::string>() + vm["output-file"].as<std::string>();
    TFile* f = TFile::Open(file.c_str(), "RECREATE");
    if (f == nullptr) {
      std::cout << "Error: File" << file << " could not be open for writing !!!" << std::endl;
      return 1;
    } else {
      std::cout << "File" << file << " being writen." << std::endl;
      f->WriteObject(&vect, "ccdb_object");
      f->Close();
    }
  } else {
    std::cout << "No file created" << std::endl;
  }
  return 0;
}
