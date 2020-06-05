// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file digi2raw.cxx
/// \author ruben.shahoyan@cern.ch

#include <boost/program_options.hpp>
#include <TSystem.h>
#include <TFile.h>
#include <TStopwatch.h>
#include "Framework/Logger.h"
#include <string>
#include <iomanip>
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsRaw/HBFUtils.h"
#include "FT0Simulation/Digits2Raw.h"
#include "DataFormatsParameters/GRPObject.h"

/// MC->raw conversion for FT0

namespace bpo = boost::program_options;

void digi2raw(const std::string& inpName, const std::string& outDir, int verbosity, bool filePerLink, uint32_t rdhV = 4, bool noEmptyHBF = false,
              int superPageSizeInB = 1024 * 1024);

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Convert FT0 digits to CRU raw data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbosity,v", bpo::value<int>()->default_value(0), "verbosity level");
    //    add_option("input-file,i", bpo::value<std::string>()->default_value(o2::base::NameConf::getDigitsFileName(o2::detectors::DetID::FT0)),"input FT0 digits file"); // why not used?
    add_option("input-file,i", bpo::value<std::string>()->default_value("ft0digits.root"), "input FT0 digits file");
    add_option("file-per-link,l", bpo::value<bool>()->default_value(false)->implicit_value(true), "create output file per CRU (default: per layer)");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    uint32_t defRDH = o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>();
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(defRDH), "RDH version to use");
    add_option("no-empty-hbf,e", bpo::value<bool>()->default_value(false)->implicit_value(true), "do not create empty HBF pages (except for HBF starting TF)");
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
  o2::conf::ConfigurableParam::updateFromString(vm["configKeyValues"].as<std::string>());
  digi2raw(vm["input-file"].as<std::string>(),
           vm["output-dir"].as<std::string>(),
           vm["verbosity"].as<int>(),
           vm["file-per-link"].as<bool>(),
           vm["rdh-version"].as<uint32_t>(),
           vm["no-empty-hbf"].as<bool>());

  return 0;
}

void digi2raw(const std::string& inpName, const std::string& outDir, int verbosity, bool filePerLink, uint32_t rdhV, bool noEmptyHBF, int superPageSizeInB)
{
  TStopwatch swTot;
  swTot.Start();
  o2::ft0::Digits2Raw m2r;
  m2r.setFilePerLink(filePerLink);
  m2r.setVerbosity(verbosity);
  auto& wr = m2r.getWriter();
  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
  wr.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::FT0)); // must be set explicitly
  wr.setSuperPageSize(superPageSizeInB);
  wr.useRDHVersion(rdhV);
  wr.setDontFillEmptyHBF(noEmptyHBF);

  std::string outDirName(outDir);
  if (outDirName.back() != '/') {
    outDirName += '/';
  }
  // if needed, create output directory
  if (gSystem->AccessPathName(outDirName.c_str())) {
    if (gSystem->mkdir(outDirName.c_str(), kTRUE)) {
      LOG(FATAL) << "could not create output directory " << outDirName;
    } else {
      LOG(INFO) << "created output directory " << outDirName;
    }
  }

  m2r.readDigits(outDirName, inpName);
  wr.writeConfFile(wr.getOrigin().str, "RAWDATA", o2::utils::concat_string(outDirName, wr.getOrigin().str, "raw.cfg"));
  //
  swTot.Stop();
  swTot.Print();
}
