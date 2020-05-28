// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   rawfileCheck.h
/// @author ruben.shahoyan@cern.ch
/// @brief  Checker for raw data conformity with CRU format

#include "DetectorsRaw/RawFileReader.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/Logger.h"
#include <TStopwatch.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace bpo = boost::program_options;

using namespace o2::raw;

int main(int argc, char* argv[])
{
  RawFileReader reader;
  std::vector<std::string> fnames;
  std::string config, configKeyValues;
  bpo::variables_map vm;
  bpo::options_description descOpt("Options");
  descOpt.add_options()(
    "help,h", "print this help message.")(
    "conf,c", bpo::value(&config)->default_value(""), "read input from configuration file")(
    "max-tf,m", bpo::value<uint32_t>()->default_value(0xffffffff), "max.number of TF to read")(
    "verbosity,v", bpo::value<int>()->default_value(reader.getVerbosity()), "1: long report, 2 or 3: print or dump all RDH")(
    "spsize,s", bpo::value<int>()->default_value(reader.getNominalSPageSize()), "nominal super-page size in bytes")(
    "buffer-size,b", bpo::value<size_t>()->default_value(reader.getNominalSPageSize()), "buffer size for files preprocessing")(
    "configKeyValues", bpo::value(&configKeyValues)->default_value(""), "semicolon separated key=value strings")(
    RawFileReader::nochk_opt(RawFileReader::ErrWrongPacketCounterIncrement).c_str(), RawFileReader::nochk_expl(RawFileReader::ErrWrongPacketCounterIncrement).c_str())(
    RawFileReader::nochk_opt(RawFileReader::ErrWrongPageCounterIncrement).c_str(), RawFileReader::nochk_expl(RawFileReader::ErrWrongPageCounterIncrement).c_str())(
    RawFileReader::nochk_opt(RawFileReader::ErrHBFStopOnFirstPage).c_str(), RawFileReader::nochk_expl(RawFileReader::ErrHBFStopOnFirstPage).c_str())(
    RawFileReader::nochk_opt(RawFileReader::ErrHBFNoStop).c_str(), RawFileReader::nochk_expl(RawFileReader::ErrHBFNoStop).c_str())(
    RawFileReader::nochk_opt(RawFileReader::ErrWrongFirstPage).c_str(), RawFileReader::nochk_expl(RawFileReader::ErrWrongFirstPage).c_str())(
    RawFileReader::nochk_opt(RawFileReader::ErrWrongHBFsPerTF).c_str(), RawFileReader::nochk_expl(RawFileReader::ErrWrongHBFsPerTF).c_str())(
    RawFileReader::nochk_opt(RawFileReader::ErrWrongNumberOfTF).c_str(), RawFileReader::nochk_expl(RawFileReader::ErrWrongNumberOfTF).c_str())(
    RawFileReader::nochk_opt(RawFileReader::ErrHBFJump).c_str(), RawFileReader::nochk_expl(RawFileReader::ErrHBFJump).c_str())(
    RawFileReader::nochk_opt(RawFileReader::ErrNoSuperPageForTF).c_str(), RawFileReader::nochk_expl(RawFileReader::ErrNoSuperPageForTF).c_str());

  bpo::options_description hiddenOpt("hidden");
  hiddenOpt.add_options()("files", bpo::value(&fnames)->composing(), "");

  bpo::options_description fullOpt("cmd");
  fullOpt.add(descOpt).add(hiddenOpt);

  bpo::positional_options_description posOpt;
  posOpt.add("files", -1);

  auto printHelp = [&](std::ostream& stream) {
    stream << "Usage:   " << argv[0] << " [options] file0 [... fileN]" << std::endl;
    stream << descOpt << std::endl;
    stream << "  (input files are optional if config file was provided)" << std::endl;
  };

  try {
    bpo::store(bpo::command_line_parser(argc, argv)
                 .options(fullOpt)
                 .positional(posOpt)
                 .allow_unregistered()
                 .run(),
               vm);
    bpo::notify(vm);
    if (argc == 1 || vm.count("help") || (fnames.empty() && config.empty())) {
      printHelp(std::cout);
      return 0;
    }
    o2::conf::ConfigurableParam::updateFromString(configKeyValues);
  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing command line arguments\n";
    printHelp(std::cerr);
    return -1;
  }

  RawFileReader::RDH rdh;
  LOG(INFO) << "RawDataHeader v" << int(rdh.version) << " is assumed";

  reader.setVerbosity(vm["verbosity"].as<int>());
  reader.setNominalSPageSize(vm["spsize"].as<int>());
  reader.setMaxTFToRead(vm["max-tf"].as<uint32_t>());
  reader.setBufferSize(vm["buffer-size"].as<size_t>());
  uint32_t errmap = 0xffffffff;
  for (int i = RawFileReader::NErrorsDefined; i--;) {
    if (vm.count(RawFileReader::nochk_opt(RawFileReader::ErrTypes(i)).c_str())) {
      errmap ^= 0x1 << i;
      LOGF(INFO, "ignore  check for /%s/", RawFileReader::ErrNames[i].data());
    }
  }

  if (!config.empty()) {
    auto inp = o2::raw::RawFileReader::parseInput(config);
    reader.loadFromInputsMap(inp);
  }

  for (int i = 0; i < fnames.size(); i++) {
    reader.addFile(fnames[i]);
  }

  TStopwatch sw;
  sw.Start();

  reader.setCheckErrors(errmap);
  reader.init();

  sw.Print();

  return 0;
}
