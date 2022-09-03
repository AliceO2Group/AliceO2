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
#include <TSystem.h>
#include "DataFormatsParameters/GRPECSObject.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CCDB/CcdbApi.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/StringUtils.h"

using DetID = o2::detectors::DetID;
using CcdbApi = o2::ccdb::CcdbApi;
using GRPECSObject = o2::parameters::GRPECSObject;
namespace bpo = boost::program_options;

enum CCDBRefreshMode { NONE,
                       ASYNC,
                       SYNC };

void createGRPECSObject(const std::string& dataPeriod,
                        int run,
                        int runType,
                        int nHBPerTF,
                        const std::string& detsReadout,
                        const std::string& detsContinuousRO,
                        const std::string& detsTrigger,
                        const std::string& flpList,
                        long tstart,
                        long tend,
                        long marginAtSOR,
                        long marginAtEOR,
                        const std::string& ccdbServer = "",
                        const std::string& metaDataStr = "",
                        CCDBRefreshMode refresh = CCDBRefreshMode::NONE)
{
  auto detMask = o2::detectors::DetID::getMask(detsReadout);
  if (detMask.count() == 0) {
    throw std::runtime_error("empty detectors list is provided");
  }
  if (runType < 0 || runType >= int(GRPECSObject::RunType::NRUNTYPES)) {
    LOGP(warning, "run type {} is not recognized, consider updating GRPECSObject.h", runType);
  }
  auto detMaskCont = detMask & o2::detectors::DetID::getMask(detsContinuousRO);
  auto detMaskTrig = detMask & o2::detectors::DetID::getMask(detsTrigger);
  auto flpsVec = o2::utils::Str::tokenize(flpList, ',');
  for (const auto& s : flpsVec) {
    try {
      addFlp((unsigned short)std::stoi(s));
    } catch (const std::exception& e) {
      LOG(alarm) << "could not convert string " << s << " to integer FLP ID, error : " << e.what();
    }
  }
  LOG(info) << tstart << " " << tend;
  if (tstart == 0) {
    tstart = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  }
  long tendVal = 0;
  if (tend < tstart) {
    tendVal = tstart + marginAtSOR;
  } else if (tendVal < tend) {
    tendVal = tend + marginAtEOR;
  }
  GRPECSObject grpecs;
  grpecs.setTimeStart(tstart);
  grpecs.setTimeEnd(tend);

  grpecs.setNHBFPerTF(nHBPerTF);
  grpecs.setDetsReadOut(detMask);
  grpecs.setDetsContinuousReadOut(detMaskCont);
  grpecs.setDetsTrigger(detMaskTrig);
  grpecs.setRun(run);
  grpecs.setRunType((GRPECSObject::RunType)runType);
  grpecs.setDataPeriod(dataPeriod);

  grpecs.print();
  std::map<std::string, std::string> metadata;

  auto toKeyValPairs = [&metadata](const std::string& str) {
    auto v0 = o2::utils::Str::tokenize(str, ';', true);
    for (auto& token : v0) {
      auto keyval = o2::utils::Str::tokenize(token, '=', false);
      if (keyval.size() != 2) {
        LOG(error) << "Illegal command-line key/value string: " << token;
        continue;
      }
      o2::utils::Str::trim(keyval[1]);
      LOGP(info, "Adding key {} with value {}", keyval[0], keyval[1]);
      metadata[keyval[0]] = keyval[1];
    }
  };

  toKeyValPairs(metaDataStr);

  if (!ccdbServer.empty()) {
    CcdbApi api;
    const std::string objPath{"GLO/Config/GRPECS"};
    api.init(ccdbServer);
    metadata["responsible"] = "ECS";
    metadata[o2::base::NameConf::CCDBRunTag.data()] = std::to_string(run);
    metadata["EOR"] = fmt::format("{}", tend);
    api.storeAsTFileAny(&grpecs, objPath, metadata, tstart, tendVal); // making it 1-year valid to be sure we have something
    LOGP(info, "Uploaded to {}/{} with validity {}:{} for SOR:{}/EOR:{}", ccdbServer, objPath, tstart, tendVal, tstart, tend);
    if (tend > tstart) {
      // override SOR version to the same limits
      metadata.erase("EOR");
      auto prevHeader = api.retrieveHeaders(objPath, metadata, tendVal + 1); // is there an object to override
      const auto itETag = prevHeader.find("ETag");
      if (itETag != prevHeader.end()) {
        std::string etag = itETag->second;
        etag.erase(remove(etag.begin(), etag.end(), '\"'), etag.end());
        LOGP(info, "Overriding run {} SOR-only version {}{}{}/{} validity to match complete SOR/EOR version validity", run, ccdbServer, ccdbServer.back() == '/' ? "" : "/", prevHeader["Valid-From"], etag);
        api.updateMetadata(objPath, {}, std::max(tstart, tendVal - 1), etag, tendVal);
      }
      if (runType == GRPECSObject::RunType::PHYSICS) { // also storing the RCT/Info/RunInformation entry in case the run type is PHYSICS and if we are at the end of run
        char tempChar{};
        std::map<std::string, std::string> mdRCT;
        mdRCT["SOR"] = std::to_string(tstart);
        mdRCT["EOR"] = std::to_string(tend);
        long startValRCT = (long)run;
        long endValRCT = (long)(run + 1);
        api.storeAsBinaryFile(&tempChar, sizeof(tempChar), "tmp.dat", "char", "RCT/Info/RunInformation", mdRCT, startValRCT, endValRCT);
        LOGP(info, "Uploaded RCT object to {}/{} with validity {}:{}", ccdbServer, "RCT/Info/RunInformation", startValRCT, endValRCT);
      }
    }

  } else { // write a local file
    auto fname = o2::base::NameConf::getGRPECSFileName();
    TFile grpF(fname.c_str(), "recreate");
    grpF.WriteObjectAny(&grpecs, grpecs.Class(), o2::base::NameConf::CCDBOBJECT.data());
    LOG(info) << "Stored to local file " << fname;
  }
  //
  if (refresh != CCDBRefreshMode::NONE && !ccdbServer.empty()) {
    auto cmd = fmt::format("curl -I -i -s \"{}{}latest/%5Cw%7B3%7D/.*/`date +%s000`/?prepare={}\"", ccdbServer, ccdbServer.back() == '/' ? "" : "/", refresh == CCDBRefreshMode::SYNC ? "sync" : "true");
    auto t0 = std::chrono::high_resolution_clock::now();
    auto res = gSystem->Exec(cmd.c_str());
    auto t1 = std::chrono::high_resolution_clock::now();
    LOGP(info, "Executed [{}] -> {} in {:.3f} s", cmd, res, std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.f);
  }
}

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general(
    "Create GRP-ECS object and upload to the CCDB\n"
    "Usage:\n  " +
    std::string(argv[0]) +
    "");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("period,p", bpo::value<std::string>(), "data taking period");
    add_option("run,r", bpo::value<int>(), "run number");
    add_option("run-type,t", bpo::value<int>()->default_value(int(GRPECSObject::RunType::NONE)), "run type");
    add_option("hbf-per-tf,n", bpo::value<int>()->default_value(128), "number of HBFs per TF");
    add_option("detectors,d", bpo::value<string>()->default_value("all"), "comma separated list of detectors");
    add_option("continuous,c", bpo::value<string>()->default_value("ITS,TPC,TOF,MFT,MCH,MID,ZDC,FT0,FV0,FDD,CTP"), "comma separated list of detectors in continuous readout mode");
    add_option("triggering,g", bpo::value<string>()->default_value("FT0,FV0"), "comma separated list of detectors providing a trigger");
    add_option("flps,f", bpo::value<string>()->default_value(""), "comma separated list of FLPs in the data taking");
    add_option("start-time,s", bpo::value<long>()->default_value(0), "run start time in ms, now() if 0");
    add_option("end-time,e", bpo::value<long>()->default_value(0), "run end time in ms, start-time+3days is used if 0");
    add_option("ccdb-server", bpo::value<std::string>()->default_value("http://alice-ccdb.cern.ch"), "CCDB server for upload, local file if empty");
    add_option("meta-data,m", bpo::value<std::string>()->default_value("")->implicit_value(""), "metadata as key1=value1;key2=value2;..");
    add_option("refresh", bpo::value<string>()->default_value("")->implicit_value("async"), R"(refresh server cache after upload: "none" (or ""), "async" (non-blocking) and "sync" (blocking))");
    add_option("marginSOR", bpo::value<long>()->default_value(4 * o2::ccdb::CcdbObjectInfo::DAY), "validity at SOR");
    add_option("marginEOR", bpo::value<long>()->default_value(10 * o2::ccdb::CcdbObjectInfo::MINUTE), "validity margin to add after EOR");
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
  if (vm.count("run") == 0) {
    std::cerr << "ERROR: "
              << "obligator run number is missing" << std::endl;
    std::cerr << opt_general << std::endl;
    exit(3);
  }
  if (vm.count("period") == 0) {
    std::cerr << "ERROR: "
              << "obligator data taking period name is missing" << std::endl;
    std::cerr << opt_general << std::endl;
    exit(3);
  }
  std::string refreshStr = vm["refresh"].as<string>();
  CCDBRefreshMode refresh = CCDBRefreshMode::NONE;
  if (!refreshStr.empty() && refreshStr != "none") {
    if (refreshStr == "async") {
      refresh = CCDBRefreshMode::ASYNC;
    } else if (refreshStr == "sync") {
      refresh = CCDBRefreshMode::SYNC;
    } else {
      LOGP(fatal, R"(Wrong CCDB refresh mode {}, supported are "none" (or ""), "async" and "sync")", refreshStr);
    }
  }

  createGRPECSObject(
    vm["period"].as<std::string>(),
    vm["run"].as<int>(),
    vm["run-type"].as<int>(),
    vm["hbf-per-tf"].as<int>(),
    vm["detectors"].as<std::string>(),
    vm["continuous"].as<std::string>(),
    vm["triggering"].as<std::string>(),
    vm["flps"].as<std::string>(),
    vm["start-time"].as<long>(),
    vm["end-time"].as<long>(),
    vm["marginSOR"].as<long>(),
    vm["marginEOR"].as<long>(),
    vm["ccdb-server"].as<std::string>(),
    vm["meta-data"].as<std::string>(),
    refresh);
}
