// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DigitTreeReader.h"
#include "Framework/Logger.h"
#include "MCHRawEncoderDigit/DigitRawEncoder.h"
#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>
#include <boost/program_options.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

/** Program to convert MCH digits to MCH Raw data.
*
* Typical usage :
*
* o2-mch-digits-to-raw --input-file mchdigits.root --file-per-link
*
* Note: the dummy-elecmap adds some complexity here,
* but is used as a mean to get the electronic mapping for the whole detector
* (as opposed to just the parts that are currently installed at Pt2).
* It will be removed when the actual electronic mapping
* (ElectronicMapperGenerated) is completed.
*
*/

namespace po = boost::program_options;
using namespace o2::mch::raw;

int main(int argc, char* argv[])
{
  po::options_description generic("options");
  std::string input;
  po::variables_map vm;

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("userLogic,u",po::bool_switch()->default_value(true),"user logic format")
      ("dummy-elecmap,d",po::bool_switch()->default_value(false),"use a dummy electronic mapping (for testing only, to be removed at some point)")
      ("output-dir,o",po::value<std::string>()->default_value("./"),"output directory for file(s)")
      ("file-per-link,l", po::value<bool>()->default_value(false)->implicit_value(true), "produce single file per link")
      ("input-file,i",po::value<std::string>(&input)->default_value("mchdigits.root"),"input file name")
      ("configKeyValues", po::value<std::string>()->default_value(""), "comma-separated configKeyValues")
      ("no-empty-hbf,e", po::value<bool>()->default_value(false), "do not create empty HBF pages (except for HBF starting TF)")
      ("raw-file-writer-verbosity,v", po::value<int>()->default_value(0), "verbosity level of the RawFileWriter")
      ("hbfutils-config", po::value<std::string>()->default_value(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)), "config file for HBFUtils (or none)")
      ("rdh-version,r", po::value<int>()->default_value(o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>()), "RDH version to use")
      ("verbosity,v",po::value<std::string>()->default_value("verylow"), "(fair)logger verbosity");
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << generic << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  po::notify(vm);

  // first things first : check the input path actually exists
  std::ifstream in(input);
  if (!in) {
    LOGF(FATAL, "could not open input file {}", input);
    exit(2);
  }

  std::string confDig = vm["hbfutils-config"].as<std::string>();
  if (!confDig.empty() && confDig != "none") {
    o2::conf::ConfigurableParam::updateFromFile(confDig, "HBFUtils");
  }
  o2::conf::ConfigurableParam::updateFromString(vm["configKeyValues"].as<std::string>());

  if (vm.count("verbosity")) {
    fair::Logger::SetVerbosity(vm["verbosity"].as<std::string>());
  }

  o2::mch::raw::DigitRawEncoderOptions opts;

  opts.noEmptyHBF = vm["no-empty-hbf"].as<bool>();
  opts.outputDir = vm["output-dir"].as<std::string>();
  opts.filePerLink = vm["file-per-link"].as<bool>();
  opts.userLogic = vm["userLogic"].as<bool>();
  opts.dummyElecMap = vm["dummy-elecmap"].as<bool>();
  opts.rawFileWriterVerbosity = vm["raw-file-writer-verbosity"].as<int>();
  opts.rdhVersion = vm["rdh-version"].as<int>();

  o2::mch::raw::DigitRawEncoder dre(opts);

  TFile fin(input.c_str());
  if (!fin.IsOpen()) {
    std::cout << "Can not open Root input file " << input << "\n";
    return -1;
  }
  TTree* tree = static_cast<TTree*>(fin.Get("o2sim"));
  if (!tree) {
    std::cout << "Can not get input tree o2sim from file " << input << "\n";
    return -2;
  }

  DigitTreeReader dr(tree);

  // here we implicitely assume that this digits-to-raw is only called for
  // one timeframe so it's easy to detect the TF start...
  uint32_t firstOrbitOfRun = o2::raw::HBFUtils::Instance().orbitFirst;
  auto dsElecIds = opts.dummyElecMap ? getAllDs<ElectronicMapperDummy>() : getAllDs<ElectronicMapperGenerated>();
  dre.addHeartbeats(dsElecIds, firstOrbitOfRun);

  o2::mch::ROFRecord rof;
  std::vector<o2::mch::Digit> digits;

  // Loop over digits (grouped per ROF) and encode them
  while (dr.nextDigits(rof, digits)) {
    if (rof.getNEntries() != digits.size()) {
      LOGP(error, "Inconsistent rof number of entries {} != number of digits {}", rof.getNEntries(), digits.size());
    }
    dre.encodeDigits(digits, rof.getBCData().orbit, rof.getBCData().bc);
  }

  // write raw files configuration so it can be used to easily read them back
  dre.writeConfig();

  return 0;
}
