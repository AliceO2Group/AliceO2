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
#include "DataFormatsParameters/GRPECSObject.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CCDB/CcdbApi.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/StringUtils.h"

using DetID = o2::detectors::DetID;
using CcdbApi = o2::ccdb::CcdbApi;
using GRPECSObject = o2::parameters::GRPECSObject;
namespace bpo = boost::program_options;

void createGRPECSObject(const std::string& dataPeriod,
                        int run,
                        int runType,
                        int nHBPerTF,
                        const std::string& detsReadout,
                        const std::string& detsContinuousRO,
                        const std::string& detsTrigger,
                        long tstart,
                        long tend,
                        std::string ccdbServer = "")
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
  LOG(info) << tstart << " " << tend;
  if (tstart == 0) {
    tstart = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }
  long tendVal = 0;
  if (tend < tstart) {
    tendVal = tstart + 2 * 24 * 3600 * 1000UL; // assume that we will never have run longer than 2 days
    LOG(info) << tstart << " -> " << tend;
  } else if (tendVal < tend) {
    tendVal = tend + 3600 * 1000UL; // we want the version with EOR to fully override the version w/o EOR
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

  if (!ccdbServer.empty()) {
    CcdbApi api;
    api.init(ccdbServer);
    std::map<std::string, std::string> metadata;
    metadata["responsible"] = "ECS";
    metadata[o2::base::NameConf::CCDBRunTag.data()] = std::to_string(run);
    // long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    api.storeAsTFileAny(&grpecs, "GLO/Config/GRPECS", metadata, tstart, tendVal); // making it 1-year valid to be sure we have something
    LOGP(info, "Uploaded to {}/{} with validity {}:{}", ccdbServer, "GLO/Config/GRPECS", tstart, tendVal);
    // also storing the RCT/Info/RunInformation entry in case the run type is PHYSICS and if we are at the end of run
    if (runType == GRPECSObject::RunType::PHYSICS && tend < tstart) {
      char tempChar;
      std::map<std::string, std::string> mdRCT;
      mdRCT["SOR"] = std::to_string(tstart);
      mdRCT["EOR"] = std::to_string(tend);
      long startValRCT = (long)run;
      long endValRCT = (long)(run + 1);
      api.storeAsBinaryFile(&tempChar, sizeof(tempChar), "tmp.dat", "char", "RCT/Info/RunInformation", mdRCT, startValRCT, endValRCT);
      LOGP(info, "Uploaded RCT object to {}/{} with validity {}:{}", ccdbServer, "RCT/Info/RunInformation", startValRCT, endValRCT);
    }
  } else { // write a local file
    auto fname = o2::base::NameConf::getGRPECSFileName();
    TFile grpF(fname.c_str(), "recreate");
    grpF.WriteObjectAny(&grpecs, grpecs.Class(), o2::base::NameConf::CCDBOBJECT.data());
    LOG(info) << "Stored to local file " << fname;
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
    add_option("start-time,s", bpo::value<long>()->default_value(0), "run start time in ms, now() if 0");
    add_option("end-time,e", bpo::value<long>()->default_value(0), "run end time in ms, start-time+3days is used if 0");
    add_option("ccdb-server", bpo::value<std::string>()->default_value("http://alice-ccdb.cern.ch"), "CCDB server for upload, local file if empty");

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

  createGRPECSObject(
    vm["period"].as<std::string>(),
    vm["run"].as<int>(),
    vm["run-type"].as<int>(),
    vm["hbf-per-tf"].as<int>(),
    vm["detectors"].as<std::string>(),
    vm["continuous"].as<std::string>(),
    vm["triggering"].as<std::string>(),
    vm["start-time"].as<long>(),
    vm["end-time"].as<long>(),
    vm["ccdb-server"].as<std::string>());
}
