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

/// \file digit2raw.cxx
/// \author ruben.shahoyan@cern.ch afurs@cern.ch

#include <boost/program_options.hpp>
#include <TStopwatch.h>
#include <string>
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtils.h"
#include "FV0Raw/RawWriterFV0.h"
#include "DataFormatsParameters/GRPObject.h"

/// MC->raw conversion for FDD

namespace bpo = boost::program_options;
struct Configuration {
  Configuration(const bpo::variables_map& vm)
  {
    mInputFile = vm["input-file"].as<std::string>();
    mOutputDir = vm["output-dir"].as<std::string>();
    mVerbosity = vm["verbosity"].as<int>();
    mFileFor = vm["file-for"].as<std::string>();
    mRdhVersion = vm["rdh-version"].as<uint32_t>();
    mEnablePadding = vm["enable-padding"].as<bool>();
    mNoEmptyHBF = vm["no-empty-hbf"].as<bool>();
    mFlpName = vm["flp-name"].as<std::string>();
    mCcdbPath = vm["ccdb-path"].as<std::string>();
    mChannelMapPath = vm["lut-path"].as<std::string>();
    if (mRdhVersion < 7 && !mEnablePadding) {
      mEnablePadding = true;
      LOG(info) << "padding is always ON for RDH version " << mRdhVersion;
    }
    mDataFormat = mEnablePadding ? 0 : 2;
  }
  bool mNoEmptyHBF;
  bool mEnablePadding;
  int mVerbosity;
  uint32_t mRdhVersion;
  uint32_t mDataFormat;
  std::string mInputFile;
  std::string mOutputDir;
  std::string mFileFor;
  std::string mFlpName;
  std::string mCcdbPath;
  std::string mChannelMapPath;
};

template <typename RawWriterType>
void digit2raw(const Configuration& cfg);

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Convert FV0 digits to CRU raw data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbosity,v", bpo::value<int>()->default_value(0), "verbosity level");
    add_option("input-file,i", bpo::value<std::string>()->default_value("fv0digits.root"), "input FV0 digits file");
    add_option("flp-name", bpo::value<std::string>()->default_value("alio2-cr1-flp180"), "single file per: all,flp,cru,link");
    add_option("file-for,f", bpo::value<std::string>()->default_value("all"), "single file per: all,flp,cruendpoint,link");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    uint32_t defRDH = o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>();
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(defRDH), "RDH version to use");
    add_option("no-empty-hbf,e", bpo::value<bool>()->default_value(false)->implicit_value(true), "do not create empty HBF pages (except for HBF starting TF)");
    add_option("hbfutils-config,u", bpo::value<std::string>()->default_value(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)), "config file for HBFUtils (or none)");
    add_option("configKeyValues", bpo::value<std::string>()->default_value(""), "comma-separated configKeyValues");
    add_option("ccdb-path", bpo::value<std::string>()->default_value(""), "CCDB url which contains LookupTable");
    add_option("lut-path", bpo::value<std::string>()->default_value(""), "LookupTable path, e.g. FV0/LookupTable");
    add_option("enable-padding", bpo::value<bool>()->default_value(false)->implicit_value(true), "enable GBT word padding to 128 bits even for RDH V7");
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

  const Configuration cfg(vm);
  if (!cfg.mEnablePadding) {
    digit2raw<o2::fv0::RawWriterFV0>(cfg);
  } else {
    digit2raw<o2::fv0::RawWriterFV0_padded>(cfg);
  }
  o2::raw::HBFUtils::Instance().print();

  return 0;
}

template <typename RawWriterType>
void digit2raw(const Configuration& cfg)
{
  TStopwatch swTot;
  swTot.Start();
  RawWriterType m2r;
  m2r.setFileFor(cfg.mFileFor);
  m2r.setFlpName(cfg.mFlpName);
  m2r.setVerbosity(cfg.mVerbosity);
  if (cfg.mCcdbPath != "") {
    m2r.setCCDBurl(cfg.mCcdbPath);
  }
  if (cfg.mChannelMapPath != "") {
    m2r.setLUTpath(cfg.mChannelMapPath);
  }
  auto& wr = m2r.getWriter();
  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
  wr.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::FV0)); // must be set explicitly
  const int superPageSizeInB = 1024 * 1024;
  wr.setSuperPageSize(superPageSizeInB);
  wr.useRDHVersion(cfg.mRdhVersion);
  wr.setDontFillEmptyHBF(cfg.mNoEmptyHBF);
  wr.useRDHDataFormat(cfg.mDataFormat);
  if (!cfg.mEnablePadding) { // CRU page alignment padding is used only if no GBT word padding is used
    wr.setAlignmentSize(16); // change to constexpr static field from class?
    wr.setAlignmentPaddingFiller(0xff);
  }
  o2::raw::assertOutputDirectory(cfg.mOutputDir);

  std::string outDirName(cfg.mOutputDir);
  if (outDirName.back() != '/') {
    outDirName += '/';
  }

  m2r.convertDigitsToRaw(outDirName, cfg.mInputFile);
  wr.writeConfFile(wr.getOrigin().str, "RAWDATA", o2::utils::Str::concat_string(outDirName, wr.getOrigin().str, "raw.cfg"));
  //
  swTot.Stop();
  swTot.Print();
}
