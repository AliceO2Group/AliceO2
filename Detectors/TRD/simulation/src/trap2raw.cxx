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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD Raw data format generation                                           //
//  Take the output of the trapsimulator the [2-4]x32 bit words and          //
//  associated headers and links etc. and produce the output of the cru.     //
//  Hence the incredibly original name.                                      //
//  Now also digits.
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "fairlogger/Logger.h"

#include "CommonUtils/StringUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/HBFUtils.h"
#include "TRDSimulation/Trap2CRU.h"
#include "DataFormatsParameters/GRPObject.h"

#include <boost/program_options.hpp>
#include <filesystem>
#include <TStopwatch.h>
#include <string>
#include <iomanip>
#include <iostream>
#include <iomanip>

namespace bpo = boost::program_options;

void trap2raw(const std::string& inpDigitsName, const std::string& inpTrackletsName,
              const std::string& outDir, int verbosity, std::string filePerLink,
              uint32_t rdhV = 6, bool noEmptyHBF = false, int tracklethcheader = 0, int superPageSizeInB = 1024 * 1024);

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Convert TRD sim otuput to CRU raw data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbosity,v", bpo::value<int>()->default_value(0), "verbosity level");
    add_option("input-file-digits,d", bpo::value<std::string>()->default_value("trddigits.root"), "input Trapsim digits file");
    add_option("input-file-tracklets,t", bpo::value<std::string>()->default_value("trdtracklets.root"), "input Trapsim tracklets file");
    add_option("file-per,l", bpo::value<std::string>()->default_value("halfcru"), "all : raw file(false), halfcru : cru end point, cru : one file per cru, sm: one file per supermodule");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    add_option("tracklethcheader,x", bpo::value<int>()->default_value(2), "include tracklet half chamber header (for run3). 0 never, 1 if there is tracklet data, 2 always");
    add_option("no-empty-hbf,e", bpo::value<bool>()->default_value(false)->implicit_value(true), "do not create empty HBF pages (except for HBF starting TF)");
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(6), "rdh version in use default");
    add_option("configKeyValues", bpo::value<std::string>()->default_value(""), "comma-separated configKeyValues");
    add_option("hbfutils-config,u", bpo::value<std::string>()->default_value(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)), "config file for HBFUtils (or none)");

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

  trap2raw(vm["input-file-digits"].as<std::string>(), vm["input-file-tracklets"].as<std::string>(), vm["output-dir"].as<std::string>(), vm["verbosity"].as<int>(), vm["file-per"].as<std::string>(), vm["rdh-version"].as<uint32_t>(), vm["no-empty-hbf"].as<bool>(), vm["tracklethcheader"].as<int>());

  return 0;
}

void trap2raw(const std::string& inpDigitsName, const std::string& inpTrackletsName, const std::string& outDir, int verbosity, std::string filePer, uint32_t rdhV, bool noEmptyHBF, int trackletHCHeader, int superPageSizeInB)
{
  TStopwatch swTot;
  swTot.Start();
  LOG(info) << "timer started";
  o2::trd::Trap2CRU mc2raw(outDir, inpDigitsName, inpTrackletsName); //,superPageSizeInB);
  LOG(info) << "class instantiated";
  mc2raw.setFilePer(filePer);
  mc2raw.setVerbosity(verbosity);
  mc2raw.setTrackletHCHeader(trackletHCHeader); // run3 or run2
  auto& wr = mc2raw.getWriter();
  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
  // wr.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::TRD)); // must be set explicitly
  wr.setContinuousReadout(false); // above should work, but I know this is correct, come back TODO
  wr.setSuperPageSize(superPageSizeInB);
  wr.useRDHVersion(rdhV);
  wr.setDontFillEmptyHBF(noEmptyHBF);

  o2::raw::assertOutputDirectory(outDir);

  std::string outDirName(outDir);
  if (outDirName.back() != '/') {
    outDirName += '/';
  }

  mc2raw.setTrackletHCHeader(trackletHCHeader);
  mc2raw.readTrapData();
  wr.writeConfFile(wr.getOrigin().str, "RAWDATA", o2::utils::Str::concat_string(outDirName, wr.getOrigin().str, "raw.cfg"));
  //
  swTot.Stop();
  swTot.Print();
}
