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

/// \file digi2raw.cxx
/// \author ruben.shahoyan@cern.ch

#include <boost/program_options.hpp>
#include <filesystem>
#include <TFile.h>
#include <TStopwatch.h>
#include "Framework/Logger.h"
#include <string>
#include <iomanip>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/HBFUtils.h"
#include "ZDCBase/Constants.h"
#include "ZDCBase/ModuleConfig.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "ZDCSimulation/Digits2Raw.h"
#include "DataFormatsParameters/GRPObject.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimConfig/DigiParams.h"

/// MC->raw conversion for ZDC

namespace bpo = boost::program_options;

void digi2raw(const std::string& inpName, const std::string& outDir, int verbosity, const std::string& fileFor, uint32_t rdhV = 7, bool enablePadding = false,
              const std::string& ccdbHost = "", int superPageSizeInB = 1024 * 1024);

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Convert ZDC digits to CRU raw data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbosity,v", bpo::value<int>()->default_value(0), "verbosity level");
    //    add_option("input-file,i", bpo::value<std::string>()->default_value(o2::base::NameConf::getDigitsFileName(o2::detectors::DetID::ZDC)),"input ZDC digits file"); // why not used?
    add_option("input-file,i", bpo::value<std::string>()->default_value("zdcdigits.root"), "input ZDC digits file");
    add_option("file-for,f", bpo::value<std::string>()->default_value("all"), "single file per: all,flp,cruendpoint,link");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    uint32_t defRDH = o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>();
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(defRDH), "RDH version to use");
    add_option("enable-padding", bpo::value<bool>()->default_value(false)->implicit_value(true), "enable GBT word padding to 128 bits even for RDH V7");
    add_option("hbfutils-config,u", bpo::value<std::string>()->default_value(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)), "config file for HBFUtils (or none)");
    add_option("configKeyValues", bpo::value<std::string>()->default_value(""), "comma-separated configKeyValues");

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
  std::string confDig = vm["hbfutils-config"].as<std::string>();
  if (!confDig.empty() && confDig != "none") {
    o2::conf::ConfigurableParam::updateFromFile(confDig, "HBFUtils");
  }
  o2::conf::ConfigurableParam::updateFromString(vm["configKeyValues"].as<std::string>());

  std::string ccdbHost = o2::base::NameConf::getCCDBServer();
  LOG(info) << "CCDB url " << ccdbHost;
  digi2raw(vm["input-file"].as<std::string>(),
           vm["output-dir"].as<std::string>(),
           vm["verbosity"].as<int>(),
           vm["file-for"].as<std::string>(),
           vm["rdh-version"].as<uint32_t>(),
           vm["enable-padding"].as<bool>(),
           ccdbHost);

  o2::raw::HBFUtils::Instance().print();

  return 0;
}

void digi2raw(const std::string& inpName, const std::string& outDir, int verbosity, const std::string& fileFor, uint32_t rdhV, bool enablePadding, const std::string& ccdbHost, int superPageSizeInB)
{
  if (rdhV < 7 && !enablePadding) {
    enablePadding = true;
    LOG(info) << "padding is always ON for RDH version " << rdhV;
  }
  //std::string ccdbHost = "http://ccdb-test.cern.ch:8080";
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(ccdbHost);
  /*long timeStamp = 0; // TIMESTAMP SHOULD NOT BE 0!
  if (timeStamp == mgr.getTimestamp()) {
    return;
  }
  mgr.setTimestamp(timeStamp);*/
  auto moduleConfig = mgr.get<o2::zdc::ModuleConfig>(o2::zdc::CCDBPathConfigModule);
  if (!moduleConfig) {
    LOG(fatal) << "Cannot module configuratio for timestamp " << mgr.getTimestamp();
    return;
  }
  LOG(info) << "Loaded module configuration for timestamp " << mgr.getTimestamp();

  auto simCondition = mgr.get<o2::zdc::SimCondition>(o2::zdc::CCDBPathConfigSim);
  if (!simCondition) {
    LOG(fatal) << "Cannot get simulation configuration for timestamp " << mgr.getTimestamp();
    return;
  }
  LOG(info) << "Loaded simulation configuration for timestamp " << mgr.getTimestamp();
  simCondition->print();

  LOG(info) << "RDHVersion " << rdhV << " padding: " << enablePadding;

  const auto* ctx = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
  const auto& bcfill = ctx->getBunchFilling();
  auto bf = ctx->getBunchFilling();
  if (verbosity > 0) {
    bf.print();
  }
  auto bp = bf.getPattern();

  TStopwatch swTot;
  swTot.Start();

  o2::zdc::Digits2Raw d2r;
  d2r.setFileFor(fileFor);
  d2r.setEnablePadding(enablePadding);
  d2r.setVerbosity(verbosity);
  auto& wr = d2r.getWriter();
  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
  wr.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::ZDC)); // must be set explicitly
  wr.setSuperPageSize(superPageSizeInB);
  wr.useRDHVersion(rdhV);
  wr.useRDHDataFormat(enablePadding ? 0 : 2);
  if (!enablePadding) { // CRU page alignment padding is used only if no GBT word padding is used
    wr.setAlignmentSize(16);
    wr.setAlignmentPaddingFiller(0xff);
  }
  o2::raw::assertOutputDirectory(outDir);

  std::string outDirName(outDir);
  if (outDirName.back() != '/') {
    outDirName += '/';
  }

  d2r.setModuleConfig(moduleConfig);
  d2r.setSimCondition(simCondition);
  d2r.emptyBunches(bp);
  d2r.setVerbosity(verbosity);
  d2r.processDigits(outDirName, inpName);
  wr.writeConfFile(wr.getOrigin().str, "RAWDATA", o2::utils::Str::concat_string(outDirName, wr.getOrigin().str, "raw.cfg"));
  //
  swTot.Stop();
  swTot.Print();
}
