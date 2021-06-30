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

#include <memory>
#include <string>
#include <vector>
#include "Framework/Logger.h"
#include <filesystem>
#include <boost/program_options.hpp>

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/StringUtils.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "PHOSBase/Geometry.h"
#include "PHOSSimulation/RawWriter.h"
#include "DetectorsCommonDataFormats/NameConf.h"

namespace bpo = boost::program_options;

int main(int argc, const char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       " <cmds/options>\n"
                                       "  Tool will encode phos raw data from input file\n"
                                       "Commands / Options");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbose,v", bpo::value<uint32_t>()->default_value(0), "Select verbosity level [0 = no output]");
    add_option("input-file,i", bpo::value<std::string>()->default_value("phosdigits.root"), "Specifies digit input file.");
    add_option("file-for,f", bpo::value<std::string>()->default_value("all"), "single file per: all,link");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    add_option("debug,d", bpo::value<uint32_t>()->default_value(0), "Select debug output level [0 = no debug output]");
    add_option("hbfutils-config,u", bpo::value<std::string>()->default_value(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)), "config file for HBFUtils (or none)");
    add_option("configKeyValues", bpo::value<std::string>()->default_value(""), "comma-separated configKeyValues");

    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help") || argc == 1) {
      std::cout << opt_general << std::endl;
      exit(0);
    }

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

  auto digitfilename = vm["input-file"].as<std::string>(),
       outputdir = vm["output-dir"].as<std::string>(),
       filefor = vm["file-for"].as<std::string>();

  // if needed, create output directory
  if (!std::filesystem::exists(outputdir)) {
    if (!std::filesystem::create_directories(outputdir)) {
      LOG(FATAL) << "could not create output directory " << outputdir;
    } else {
      LOG(INFO) << "created output directory " << outputdir;
    }
  }

  std::unique_ptr<TFile> digitfile(TFile::Open(digitfilename.data(), "READ"));
  auto treereader = std::make_unique<TTreeReader>(static_cast<TTree*>(digitfile->Get("o2sim")));
  TTreeReaderValue<std::vector<o2::phos::Digit>> digitbranch(*treereader, "PHOSDigit");
  TTreeReaderValue<std::vector<o2::phos::TriggerRecord>> triggerbranch(*treereader, "PHOSDigitTrigRecords");

  o2::phos::RawWriter::FileFor_t granularity = o2::phos::RawWriter::FileFor_t::kFullDet;
  if (filefor == "all") {
    granularity = o2::phos::RawWriter::FileFor_t::kFullDet;
  } else if (filefor == "link") {
    granularity = o2::phos::RawWriter::FileFor_t::kLink;
  }

  o2::phos::RawWriter rawwriter;
  rawwriter.setOutputLocation(outputdir.data());
  rawwriter.setFileFor(granularity);
  rawwriter.init();

  // Loop over all entries in the tree, where each tree entry corresponds to a time frame
  for (auto en : *treereader) {
    rawwriter.digitsToRaw(*digitbranch, *triggerbranch);
  }
  rawwriter.getWriter().writeConfFile("PHS", "RAWDATA", o2::utils::Str::concat_string(outputdir, "/PHSraw.cfg"));

  o2::raw::HBFUtils::Instance().print();

  return 0;
}
