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
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CTPSimulation/Digits2Raw.h"
#include "DataFormatsParameters/GRPObject.h"
namespace bpo = boost::program_options;

void digi2raw(const std::string& inpName, const std::string& outDir, int verbosity, const std::string& fileForLink, uint32_t rdhV = 4, bool noEmptyHBF = false,
              bool zsIR = true, bool zsClass = true, int superPageSizeInB = 1024 * 1024);

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Convert CTP digits to CRU raw data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbosity,v", bpo::value<int>()->default_value(0), "verbosity level");
    //    add_option("input-file,i", bpo::value<std::string>()->default_value(o2::base::NameConf::getDigitsFileName(o2::detectors::DetID::CTP)),"input CTP digits file"); // why not used?
    add_option("input-file,i", bpo::value<std::string>()->default_value("ctpdigits.root"), "input CTP digits file");
    add_option("file-for,f", bpo::value<std::string>()->default_value("all"), "single file per: all,link,cruendpoint");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    uint32_t defRDH = o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>();
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(defRDH), "RDH version to use");
    add_option("no-empty-hbf,e", bpo::value<bool>()->default_value(false)->implicit_value(true), "do not create empty HBF pages (except for HBF starting TF)");
    add_option("no-zs-ir", bpo::value<bool>()->default_value(false)->implicit_value(true), "do not zero-suppress interaction records");
    add_option("no-zs-class", bpo::value<bool>()->default_value(false)->implicit_value(true), "do not zero-suppress trigger class records");
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
  digi2raw(vm["input-file"].as<std::string>(),
           vm["output-dir"].as<std::string>(),
           vm["verbosity"].as<int>(),
           vm["file-for"].as<std::string>(),
           vm["rdh-version"].as<uint32_t>(),
           vm["no-empty-hbf"].as<bool>(),
           !vm["no-zs-ir"].as<bool>(),
           !vm["no-zs-class"].as<bool>());

  o2::raw::HBFUtils::Instance().print();

  return 0;
}

void digi2raw(const std::string& inpName, const std::string& outDir, int verbosity, const std::string& fileForLink, uint32_t rdhV, bool noEmptyHBF, bool zsIR, bool zsClass, int superPageSizeInB)
{
  TStopwatch swTot;
  swTot.Start();
  o2::ctp::Digits2Raw m2r;
  m2r.setFilePerLink(fileForLink == "link");
  m2r.setVerbosity(verbosity);
  auto& wr = m2r.getWriter();
  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
  wr.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::CTP)); // must be set explicitly
  wr.setSuperPageSize(superPageSizeInB);
  wr.useRDHVersion(rdhV);
  wr.setDontFillEmptyHBF(noEmptyHBF);

  std::string outDirName(outDir);
  if (outDirName.back() != '/') {
    outDirName += '/';
  }
  // if needed, create output directory
  if (!std::filesystem::exists(outDirName)) {
    if (!std::filesystem::create_directories(outDirName)) {
      LOG(fatal) << "could not create output directory " << outDirName;
    } else {
      LOG(info) << "created output directory " << outDirName;
    }
  }
  m2r.setOutDir(outDirName);
  m2r.setZeroSuppressedIntRec(zsIR);
  m2r.setZeroSuppressedClassRec(zsClass);
  m2r.init();
  m2r.processDigits(inpName);
  wr.writeConfFile(wr.getOrigin().str, "RAWDATA", o2::utils::Str::concat_string(outDirName, wr.getOrigin().str, "raw.cfg"));
  //
  swTot.Stop();
  swTot.Print();
}
