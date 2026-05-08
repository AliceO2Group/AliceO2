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
#include "CTPWorkflowScalers/ctpCCDBManager.h"
#include "BookkeepingApi/BkpClientFactory.h"
#include "BookkeepingApi/BkpClient.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
namespace bpo = boost::program_options;
//
// Test in the lab
// o2-ctp-bk-write -r 37 -s 1 -c 1 --ccdb='http://acsl-ccdb.cern.ch:8083' -b 'acsl-aliecs.cern.ch:4001' -t 1753185071753
//
int main(int argc, char** argv)
{
  const std::string testCCDB = "http://ccdb-test.cern.ch:8080";
  // std::string prodCCDB = "http://o2-ccdb.internal";
  const std::string aliceCCDB = "http://alice-ccdb.cern.ch";
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       " Write ctp config or scalers to BK\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;
  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("input-file,f", bpo::value<std::string>()->default_value("none"), "input file name, none - do not read file");
    add_option("bkhost,b", bpo::value<std::string>()->default_value("none"), "bk web address");
    add_option("ccdb", bpo::value<std::string>()->default_value("alice"), "choose databse: test- test ccdb; prod - production ccdb; alice - alice ccdb; else ccdb parameter");
    add_option("run-number,r", bpo::value<uint32_t>()->default_value(0), "run number");
    add_option("timestamp,t", bpo::value<uint64_t>()->default_value(0), "timestamp; if 0 timestamp is calulated inside this code");
    add_option("cfg,c", bpo::value<bool>()->default_value(0), "Do cfg");
    add_option("scalers,s", bpo::value<bool>()->default_value(0), "Do scalers");
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
  uint64_t timestamp = vm["timestamp"].as<uint64_t>();
  //
  int ret = 0;
  std::vector<std::string> runs;
  int32_t run = vm["run-number"].as<uint32_t>();
  std::cout << "run:" << run << std::endl;
  if (run) {
    std::cout << "pushing" << std::endl;
    runs.push_back(std::to_string(run));
  }
  // read input file
  std::string filename = vm["input-file"].as<std::string>();
  if (filename != "none") {
    std::ifstream file(filename);
    if (!file.is_open()) {
      LOG(fatal) << "Cannot open file:" << filename << std::endl;
    } else {
      std::string line;
      while (std::getline(file, line)) {
        std::cout << line << "\n";
        std::vector<std::string> tokens = o2::utils::Str::tokenize(line, ' ');
        // int run = std::stoi(tokens[0]);
        runs.push_back(tokens[0]);
      }
    }
  }
  bool cfg = vm["cfg"].as<bool>();
  bool scalers = vm["scalers"].as<bool>();
  std::cout << "Doing: cfg:" << cfg << " scal:" << scalers << std::endl;
  if (cfg || scalers) {
    std::string bkhost = vm["bkhost"].as<std::string>();
    std::unique_ptr<o2::bkp::api::BkpClient> mBKClient = o2::bkp::api::BkpClientFactory::create(bkhost);
    // get from ccdb
    std::string ccdbAddress;
    if (vm["ccdb"].as<std::string>() == "prod") {
      // ccdbAddress = prodCCDB;
    } else if (vm["ccdb"].as<std::string>() == "test") {
      ccdbAddress = testCCDB;
    } else if (vm["ccdb"].as<std::string>() == "alice") {
      ccdbAddress = aliceCCDB;
    } else {
      ccdbAddress = vm["ccdb"].as<std::string>();
    }
    o2::ctp::ctpCCDBManager::setCCDBHost(ccdbAddress);
    std::cout << "CCDB: " << vm["ccdb"].as<std::string>() << " " << ccdbAddress << std::endl;
    std::map<std::string, std::string> metadata;
    for (auto const& run : runs) {
      metadata["runNumber"] = run;
      bool ok;
      int runNumber = std::stoi(run);
      auto ctpcfg = o2::ctp::ctpCCDBManager::getConfigFromCCDB(timestamp, run, ok);

      if (cfg) {
        std::string ctpcfgstr = ctpcfg.getConfigString();
        try {
          mBKClient->run()->setRawCtpTriggerConfiguration(runNumber, ctpcfgstr);
        } catch (std::runtime_error& error) {
          std::cerr << "An error occurred: " << error.what() << std::endl;
          // return 1;
        }
        LOG(info) << "Run BK:" << run << " CFG:" << cfg;
      }
      if (scalers) {
        auto ctpcnts = o2::ctp::ctpCCDBManager::getScalersFromCCDB(timestamp, run, "CTP/Calib/Scalers", ok);
        ctpcnts.convertRawToO2();
        std::vector<uint32_t> clsinds = ctpcnts.getClassIndexes();
        long ts = ctpcnts.getTimeLimit().second;
        int i = 0;
        for (auto const& ind : clsinds) {
          std::array<uint64_t, 7> cntsbk = ctpcnts.getIntegralForClass(i);
          std::string clsname = ctpcfg.getClassNameFromHWIndex(cntsbk[0]);
          try {
            mBKClient->ctpTriggerCounters()->createOrUpdateForRun(runNumber, clsname, ts, cntsbk[1], cntsbk[2], cntsbk[3], cntsbk[4], cntsbk[5], cntsbk[6]);
            std::cout << runNumber << " clsname: " << cntsbk[0] << " " << clsname << " t:" << ts << " cnts:" << cntsbk[1] << " " << cntsbk[2] << " " << cntsbk[3] << " " << cntsbk[4] << " " << cntsbk[5] << " " << cntsbk[6] << std::endl;
            ;

          } catch (std::runtime_error& error) {
            std::cerr << "An error occurred: " << error.what() << std::endl;
            // return 1;
          }
          LOG(debug) << "Run BK scalers ok";
          i++;
        }
      }
    }
    // add to bk
  }
  std::cout << "o2-ctp-bk-write done" << std::endl;
  return ret;
}
