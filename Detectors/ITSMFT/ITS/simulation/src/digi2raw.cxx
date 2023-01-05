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
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <TStopwatch.h>
#include "Framework/Logger.h"
#include <filesystem>
#include <vector>
#include <string>
#include <iomanip>
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITSMFTSimulation/MC2RawEncoder.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtils.h"

/// MC->raw conversion with new (variable page size) format for ITS
using MAP = o2::itsmft::ChipMappingITS;
namespace bpo = boost::program_options;

constexpr int DefRDHVersion = o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>();

void setupLinks(o2::itsmft::MC2RawEncoder<MAP>& m2r, std::string_view outDir, std::string_view outPrefix, std::string_view fileFor);
void digi2raw(std::string_view inpName, std::string_view outDir, std::string_view fileFor, int verbosity,
              uint32_t rdhV = DefRDHVersion, bool noEmptyHBF = false,
              int superPageSizeInB = 1024 * 1024);

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Convert ITS digits to CRU raw data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbosity,v", bpo::value<uint32_t>()->default_value(0), "verbosity level [0 = no output]");
    add_option("input-file,i", bpo::value<std::string>()->default_value("itsdigits.root"), "input ITS digits file");
    add_option("file-for,f", bpo::value<std::string>()->default_value("all"), "single file per: all,flp,cruendpoint,link");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(DefRDHVersion), "RDH version to use");
    add_option("no-empty-hbf,e", bpo::value<bool>()->default_value(false)->implicit_value(true), "do not create empty HBF pages (except for HBF starting TF)");
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
           vm["file-for"].as<std::string>(),
           vm["verbosity"].as<uint32_t>(),
           vm["rdh-version"].as<uint32_t>(),
           vm["no-empty-hbf"].as<bool>());
  LOG(info) << "HBFUtils settings used for conversion:";

  o2::raw::HBFUtils::Instance().print();

  return 0;
}

void digi2raw(std::string_view inpName, std::string_view outDir, std::string_view fileFor, int verbosity, uint32_t rdhV, bool noEmptyHBF, int superPageSizeInB)
{
  TStopwatch swTot;
  swTot.Start();
  using ROFR = o2::itsmft::ROFRecord;
  using ROFRVEC = std::vector<o2::itsmft::ROFRecord>;
  const uint8_t ruSWMin = 0, ruSWMax = 0xff; // seq.ID of 1st and last RU (stave) to convert

  LOG(info) << "HBFUtil settings:";
  o2::raw::HBFUtils::Instance().print();

  // if needed, create output directory
  if (!std::filesystem::exists(outDir)) {
    if (!std::filesystem::create_directories(outDir)) {
      LOG(fatal) << "could not create output directory " << outDir;
    } else {
      LOG(info) << "created output directory " << outDir;
    }
  }

  ///-------> input
  std::string digTreeName{o2::base::NameConf::MCTTREENAME.data()};
  TChain digTree(digTreeName.c_str());
  digTree.AddFile(inpName.data());
  digTree.SetBranchStatus("*MCTruth*", 0); // ignore MC info

  std::vector<o2::itsmft::Digit> digiVec, *digiVecP = &digiVec;
  std::string digBranchName = o2::utils::Str::concat_string(MAP::getName(), "Digit");
  if (!digTree.GetBranch(digBranchName.c_str())) {
    LOG(fatal) << "Failed to find the branch " << digBranchName << " in the tree " << digTreeName;
  }
  digTree.SetBranchAddress(digBranchName.c_str(), &digiVecP);

  // ROF record entries in the digit tree
  ROFRVEC rofRecVec, *rofRecVecP = &rofRecVec;
  std::string rofRecName = o2::utils::Str::concat_string(MAP::getName(), "DigitROF");
  if (!digTree.GetBranch(rofRecName.c_str())) {
    LOG(fatal) << "Failed to find the branch " << rofRecName << " in the tree " << digTreeName;
  }
  digTree.SetBranchAddress(rofRecName.c_str(), &rofRecVecP);
  ///-------< input
  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);

  o2::itsmft::MC2RawEncoder<MAP> m2r;
  m2r.setVerbosity(verbosity);
  m2r.setContinuousReadout(grp->isDetContinuousReadOut(MAP::getDetID())); // must be set explicitly
  m2r.setDefaultSinkName(o2::utils::Str::concat_string(MAP::getName(), ".raw"));
  m2r.setMinMaxRUSW(ruSWMin, ruSWMax);
  m2r.getWriter().setSuperPageSize(superPageSizeInB);
  m2r.getWriter().useRDHVersion(rdhV);
  m2r.getWriter().setDontFillEmptyHBF(noEmptyHBF);

  m2r.setVerbosity(verbosity);
  setupLinks(m2r, outDir, MAP::getName(), fileFor);
  //-------------------------------------------------------------------------------<<<<
  long nEntProc = 0;
  for (int i = 0; i < digTree.GetEntries(); i++) {
    digTree.GetEntry(i);
    for (const auto& rofRec : rofRecVec) {
      int nDigROF = rofRec.getNEntries();
      if (verbosity) {
        LOG(info) << "Processing ROF:" << rofRec.getROFrame() << " with " << nDigROF << " entries";
        rofRec.print();
      }
      if (!nDigROF) {
        if (verbosity) {
          LOG(info) << "Frame is empty"; // ??
        }
        continue;
      }
      nEntProc++;
      auto dgs = nDigROF ? gsl::span<const o2::itsmft::Digit>(&digiVec[rofRec.getFirstEntry()], nDigROF) : gsl::span<const o2::itsmft::Digit>();
      m2r.digits2raw(dgs, rofRec.getBCData());
    }
  } // loop over multiple ROFvectors (in case of chaining)

  m2r.getWriter().writeConfFile(MAP::getName(), "RAWDATA", o2::utils::Str::concat_string(outDir, '/', MAP::getName(), "raw.cfg"));
  m2r.finalize(); // finish TF and flush data
  //
  swTot.Stop();
  swTot.Print();
}

struct ITSRUMapping {
  std::string flp{};
  int cruHWID = 0;
  int layer = 0;
  int gbtChannel = 0;
  int ruHWID = 0;
  int ruInLayer = 0;
};

// FLP, CRU_HW, Lr, GBT, RU_HW, RUID_in_layer
const ITSRUMapping itsHWMap[o2::itsmft::ChipMappingITS::getNRUs()] =
  {
    {"alio2-cr1-flp187", 183, 0, 0, 28, 0},
    {"alio2-cr1-flp187", 183, 0, 1, 42, 1},
    {"alio2-cr1-flp187", 183, 0, 2, 122, 2},
    {"alio2-cr1-flp187", 183, 0, 3, 94, 3},
    {"alio2-cr1-flp187", 183, 0, 4, 49, 4},
    {"alio2-cr1-flp187", 183, 0, 5, 52, 5},
    {"alio2-cr1-flp198", 172, 0, 0, 187, 6},
    {"alio2-cr1-flp198", 172, 0, 1, 90, 7},
    {"alio2-cr1-flp198", 172, 0, 2, 102, 8},
    {"alio2-cr1-flp198", 172, 0, 3, 134, 9},
    {"alio2-cr1-flp198", 172, 0, 4, 127, 10},
    {"alio2-cr1-flp198", 172, 0, 5, 259, 11},
    {"alio2-cr1-flp188", 181, 1, 0, 54, 0},
    {"alio2-cr1-flp188", 181, 1, 1, 59, 1},
    {"alio2-cr1-flp188", 181, 1, 2, 61, 2},
    {"alio2-cr1-flp188", 181, 1, 3, 62, 3},
    {"alio2-cr1-flp188", 181, 1, 4, 167, 4},
    {"alio2-cr1-flp188", 181, 1, 5, 66, 5},
    {"alio2-cr1-flp188", 181, 1, 6, 64, 6},
    {"alio2-cr1-flp188", 181, 1, 7, 120, 7},
    {"alio2-cr1-flp203", 196, 1, 0, 199, 8},
    {"alio2-cr1-flp203", 196, 1, 1, 201, 9},
    {"alio2-cr1-flp203", 196, 1, 2, 212, 10},
    {"alio2-cr1-flp203", 196, 1, 3, 217, 11},
    {"alio2-cr1-flp203", 196, 1, 4, 230, 12},
    {"alio2-cr1-flp203", 196, 1, 5, 242, 13},
    {"alio2-cr1-flp203", 196, 1, 6, 244, 14},
    {"alio2-cr1-flp203", 196, 1, 7, 250, 15},
    {"alio2-cr1-flp189", 184, 2, 0, 63, 0},
    {"alio2-cr1-flp189", 184, 2, 1, 58, 1},
    {"alio2-cr1-flp189", 184, 2, 2, 44, 2},
    {"alio2-cr1-flp189", 184, 2, 3, 46, 3},
    {"alio2-cr1-flp189", 184, 2, 4, 50, 4},
    {"alio2-cr1-flp189", 191, 2, 0, 51, 5},
    {"alio2-cr1-flp189", 191, 2, 1, 219, 6},
    {"alio2-cr1-flp189", 191, 2, 2, 21, 7},
    {"alio2-cr1-flp189", 191, 2, 3, 29, 8},
    {"alio2-cr1-flp189", 191, 2, 4, 35, 9},
    {"alio2-cr1-flp190", 179, 2, 0, 99, 10},
    {"alio2-cr1-flp190", 179, 2, 1, 93, 11},
    {"alio2-cr1-flp190", 179, 2, 2, 97, 12},
    {"alio2-cr1-flp190", 179, 2, 3, 45, 13},
    {"alio2-cr1-flp190", 179, 2, 4, 92, 14},
    {"alio2-cr1-flp190", 192, 2, 0, 96, 15},
    {"alio2-cr1-flp190", 192, 2, 1, 125, 16},
    {"alio2-cr1-flp190", 192, 2, 2, 169, 17},
    {"alio2-cr1-flp190", 192, 2, 3, 168, 18},
    {"alio2-cr1-flp190", 192, 2, 4, 188, 19},
    {"alio2-cr1-flp191", 175, 3, 0, 106, 0},
    {"alio2-cr1-flp191", 175, 3, 1, 304, 1},
    {"alio2-cr1-flp191", 175, 3, 2, 147, 2},
    {"alio2-cr1-flp191", 175, 3, 3, 222, 3},
    {"alio2-cr1-flp191", 175, 3, 4, 293, 4},
    {"alio2-cr1-flp191", 175, 3, 5, 200, 5},
    {"alio2-cr1-flp191", 175, 3, 6, 233, 6},
    {"alio2-cr1-flp191", 175, 3, 7, 43, 7},
    {"alio2-cr1-flp191", 175, 3, 8, 173, 8},
    {"alio2-cr1-flp191", 175, 3, 9, 172, 9},
    {"alio2-cr1-flp191", 175, 3, 10, 177, 10},
    {"alio2-cr1-flp191", 175, 3, 11, 175, 11},
    {"alio2-cr1-flp191", 182, 3, 0, 2, 12},
    {"alio2-cr1-flp191", 182, 3, 1, 215, 13},
    {"alio2-cr1-flp191", 182, 3, 2, 108, 14},
    {"alio2-cr1-flp191", 182, 3, 3, 265, 15},
    {"alio2-cr1-flp191", 182, 3, 4, 241, 16},
    {"alio2-cr1-flp191", 182, 3, 5, 53, 17},
    {"alio2-cr1-flp191", 182, 3, 6, 183, 18},
    {"alio2-cr1-flp191", 182, 3, 7, 7, 19},
    {"alio2-cr1-flp191", 182, 3, 8, 191, 20},
    {"alio2-cr1-flp191", 182, 3, 9, 190, 21},
    {"alio2-cr1-flp191", 182, 3, 10, 284, 22},
    {"alio2-cr1-flp191", 182, 3, 11, 299, 23},
    {"alio2-cr1-flp192", 187, 4, 0, 273, 0},
    {"alio2-cr1-flp192", 187, 4, 1, 171, 1},
    {"alio2-cr1-flp192", 187, 4, 2, 252, 2},
    {"alio2-cr1-flp192", 187, 4, 3, 251, 3},
    {"alio2-cr1-flp192", 187, 4, 4, 202, 4},
    {"alio2-cr1-flp192", 187, 4, 5, 282, 5},
    {"alio2-cr1-flp192", 187, 4, 6, 181, 6},
    {"alio2-cr1-flp192", 187, 4, 7, 300, 7},
    {"alio2-cr1-flp192", 176, 4, 0, 302, 8},
    {"alio2-cr1-flp192", 176, 4, 1, 309, 9},
    {"alio2-cr1-flp192", 176, 4, 2, 270, 10},
    {"alio2-cr1-flp192", 176, 4, 3, 255, 11},
    {"alio2-cr1-flp192", 176, 4, 4, 203, 12},
    {"alio2-cr1-flp192", 176, 4, 5, 208, 13},
    {"alio2-cr1-flp192", 176, 4, 6, 277, 14},
    {"alio2-cr1-flp193", 177, 4, 0, 105, 23},
    {"alio2-cr1-flp193", 177, 4, 1, 258, 24},
    {"alio2-cr1-flp193", 177, 4, 2, 121, 25},
    {"alio2-cr1-flp193", 177, 4, 3, 119, 26},
    {"alio2-cr1-flp193", 177, 4, 4, 116, 27},
    {"alio2-cr1-flp193", 177, 4, 5, 135, 28},
    {"alio2-cr1-flp193", 177, 4, 6, 126, 29},
    {"alio2-cr1-flp193", 178, 4, 0, 137, 15},
    {"alio2-cr1-flp193", 178, 4, 1, 229, 16},
    {"alio2-cr1-flp193", 178, 4, 2, 272, 17},
    {"alio2-cr1-flp193", 178, 4, 3, 148, 18},
    {"alio2-cr1-flp193", 178, 4, 4, 297, 19},
    {"alio2-cr1-flp193", 178, 4, 5, 253, 20},
    {"alio2-cr1-flp193", 178, 4, 6, 84, 21},
    {"alio2-cr1-flp193", 178, 4, 7, 279, 22},
    {"alio2-cr1-flp194", 194, 5, 0, 132, 0},
    {"alio2-cr1-flp194", 194, 5, 1, 225, 1},
    {"alio2-cr1-flp194", 194, 5, 2, 240, 2},
    {"alio2-cr1-flp194", 194, 5, 3, 266, 3},
    {"alio2-cr1-flp194", 194, 5, 4, 128, 4},
    {"alio2-cr1-flp194", 194, 5, 5, 123, 5},
    {"alio2-cr1-flp194", 194, 5, 6, 170, 6},
    {"alio2-cr1-flp194", 194, 5, 7, 234, 7},
    {"alio2-cr1-flp194", 194, 5, 8, 320, 8},
    {"alio2-cr1-flp194", 194, 5, 9, 186, 9},
    {"alio2-cr1-flp194", 174, 5, 0, 245, 10},
    {"alio2-cr1-flp194", 174, 5, 1, 192, 11},
    {"alio2-cr1-flp194", 174, 5, 2, 206, 12},
    {"alio2-cr1-flp194", 174, 5, 3, 189, 13},
    {"alio2-cr1-flp194", 174, 5, 4, 213, 14},
    {"alio2-cr1-flp194", 174, 5, 5, 6, 15},
    {"alio2-cr1-flp194", 174, 5, 6, 228, 16},
    {"alio2-cr1-flp194", 174, 5, 7, 136, 17},
    {"alio2-cr1-flp194", 174, 5, 8, 197, 18},
    {"alio2-cr1-flp194", 174, 5, 9, 82, 19},
    {"alio2-cr1-flp194", 174, 5, 10, 100, 20},
    {"alio2-cr1-flp195", 180, 5, 0, 246, 31},
    {"alio2-cr1-flp195", 180, 5, 1, 271, 32},
    {"alio2-cr1-flp195", 180, 5, 2, 281, 33},
    {"alio2-cr1-flp195", 180, 5, 3, 285, 34},
    {"alio2-cr1-flp195", 180, 5, 4, 287, 35},
    {"alio2-cr1-flp195", 180, 5, 5, 289, 36},
    {"alio2-cr1-flp195", 180, 5, 6, 113, 37},
    {"alio2-cr1-flp195", 180, 5, 7, 193, 38},
    {"alio2-cr1-flp195", 180, 5, 8, 194, 39},
    {"alio2-cr1-flp195", 180, 5, 9, 195, 40},
    {"alio2-cr1-flp195", 180, 5, 10, 198, 41},
    {"alio2-cr1-flp195", 193, 5, 0, 214, 21},
    {"alio2-cr1-flp195", 193, 5, 1, 207, 22},
    {"alio2-cr1-flp195", 193, 5, 2, 248, 23},
    {"alio2-cr1-flp195", 193, 5, 3, 262, 24},
    {"alio2-cr1-flp195", 193, 5, 4, 263, 25},
    {"alio2-cr1-flp195", 193, 5, 5, 65, 26},
    {"alio2-cr1-flp195", 193, 5, 6, 56, 27},
    {"alio2-cr1-flp195", 193, 5, 7, 1, 28},
    {"alio2-cr1-flp195", 193, 5, 8, 210, 29},
    {"alio2-cr1-flp195", 193, 5, 9, 247, 30},
    {"alio2-cr1-flp196", 185, 6, 0, 36, 0},
    {"alio2-cr1-flp196", 185, 6, 1, 60, 1},
    {"alio2-cr1-flp196", 185, 6, 2, 41, 2},
    {"alio2-cr1-flp196", 185, 6, 3, 40, 3},
    {"alio2-cr1-flp196", 185, 6, 4, 80, 4},
    {"alio2-cr1-flp196", 185, 6, 5, 57, 5},
    {"alio2-cr1-flp196", 185, 6, 6, 185, 6},
    {"alio2-cr1-flp196", 185, 6, 7, 79, 7},
    {"alio2-cr1-flp196", 185, 6, 8, 91, 8},
    {"alio2-cr1-flp196", 185, 6, 9, 78, 9},
    {"alio2-cr1-flp196", 185, 6, 10, 5, 10},
    {"alio2-cr1-flp196", 185, 6, 11, 306, 11},
    {"alio2-cr1-flp196", 189, 6, 0, 39, 12},
    {"alio2-cr1-flp196", 189, 6, 1, 32, 13},
    {"alio2-cr1-flp196", 189, 6, 2, 23, 14},
    {"alio2-cr1-flp196", 189, 6, 3, 24, 15},
    {"alio2-cr1-flp196", 189, 6, 4, 22, 16},
    {"alio2-cr1-flp196", 189, 6, 5, 88, 17},
    {"alio2-cr1-flp196", 189, 6, 6, 25, 18},
    {"alio2-cr1-flp196", 189, 6, 7, 89, 19},
    {"alio2-cr1-flp196", 189, 6, 8, 87, 20},
    {"alio2-cr1-flp196", 189, 6, 9, 47, 21},
    {"alio2-cr1-flp196", 189, 6, 10, 17, 22},
    {"alio2-cr1-flp196", 189, 6, 11, 33, 23},
    {"alio2-cr1-flp197", 186, 6, 0, 180, 36},
    {"alio2-cr1-flp197", 186, 6, 1, 274, 37},
    {"alio2-cr1-flp197", 186, 6, 2, 275, 38},
    {"alio2-cr1-flp197", 186, 6, 3, 278, 39},
    {"alio2-cr1-flp197", 186, 6, 4, 276, 40},
    {"alio2-cr1-flp197", 186, 6, 5, 160, 41},
    {"alio2-cr1-flp197", 186, 6, 6, 280, 42},
    {"alio2-cr1-flp197", 186, 6, 7, 3, 43},
    {"alio2-cr1-flp197", 186, 6, 8, 209, 44},
    {"alio2-cr1-flp197", 186, 6, 9, 227, 45},
    {"alio2-cr1-flp197", 186, 6, 10, 256, 46},
    {"alio2-cr1-flp197", 186, 6, 11, 15, 47},
    {"alio2-cr1-flp197", 195, 6, 0, 115, 24},
    {"alio2-cr1-flp197", 195, 6, 1, 283, 25},
    {"alio2-cr1-flp197", 195, 6, 2, 104, 26},
    {"alio2-cr1-flp197", 195, 6, 3, 290, 27},
    {"alio2-cr1-flp197", 195, 6, 4, 254, 28},
    {"alio2-cr1-flp197", 195, 6, 5, 110, 29},
    {"alio2-cr1-flp197", 195, 6, 6, 103, 30},
    {"alio2-cr1-flp197", 195, 6, 7, 286, 31},
    {"alio2-cr1-flp197", 195, 6, 8, 257, 32},
    {"alio2-cr1-flp197", 195, 6, 9, 174, 33},
    {"alio2-cr1-flp197", 195, 6, 10, 13, 34},
    {"alio2-cr1-flp197", 195, 6, 11, 288, 35}};

void setupLinks(o2::itsmft::MC2RawEncoder<MAP>& m2r, std::string_view outDir, std::string_view outPrefix, std::string_view fileFor)
{
  //------------------------------------------------------------------------------->>>>
  // just as an example, we require here that the staves are read via 3 links, with partitioning according to lnkXB below
  // while OB staves use only 1 link.
  // Note, that if the RU container is not defined, it will be created automatically
  // during encoding.
  // If the links of the container are not defined, a single link readout will be assigned

  const auto& mp = m2r.getMapping();
  constexpr int MaxLinksPerRU = 3;
  auto getNLinks = [](int lr) { return lr < 3 ? MaxLinksPerRU : MaxLinksPerRU - 1; };
  int lnkAssign[3][MaxLinksPerRU] = {
    // requested link cabling for IB, MB and OB
    {3, 3, 3},   // IB
    {14, 14, 0}, // MB
    {14, 14, 0}  // OB
  };
  std::unordered_map<int, int> cruMaxRU, cruNRU;
  std::array<int, o2::itsmft::ChipMappingITS::getNRUs()> ruSWEntry{};
  ruSWEntry.fill(-1);
  int ntab = sizeof(itsHWMap) / sizeof(ITSRUMapping);
  for (int ir = 0; ir < ntab; ir++) {
    const auto& ru = itsHWMap[ir];
    cruMaxRU[ru.cruHWID]++;
    ruSWEntry[mp.getRUIDSW(ru.layer, ru.ruInLayer)] = ir;
  }
  std::string outFileLink;
  for (int ruID = 0; ruID < o2::itsmft::ChipMappingITS::getNRUs(); ruID++) {
    const auto& ruhw = itsHWMap[ruSWEntry[ruID]];
    int nRUsOnCRU = cruMaxRU[ruhw.cruHWID], ruOnCRU = cruNRU[ruhw.cruHWID]++;
    if (ruID < m2r.getRUSWMin() || ruID > m2r.getRUSWMax()) { // ignored RUs ?
      continue;
    }
    m2r.getCreateRUDecode(ruID); // create RU container
    int* lnkAs = lnkAssign[mp.getRUType(ruID)];
    int accL = 0;
    for (int il = 0; il < getNLinks(ruhw.layer); il++) { // create links
      auto& ru = *m2r.getRUDecode(ruID);
      ru.links[il] = m2r.addGBTLink();
      uint32_t lanes = mp.getCablesOnRUType(mp.getRUType(ruID)); // lanes pattern of this RU
      auto link = m2r.getGBTLink(ru.links[il]);
      link->lanes = lanes & (((0x1 << lnkAs[il]) - 1) << (accL));

      link->idInCRU = ruOnCRU + il * nRUsOnCRU; // linkID
      link->cruID = ruhw.cruHWID;
      link->feeID = mp.RUSW2FEEId(ruID, il);
      link->endPointID = link->idInCRU > 11 ? 1 : 0;
      accL += lnkAs[il];
      outFileLink = o2::utils::Str::concat_string(outDir, "/", outPrefix);
      if (fileFor != "all") { // single file for all links
        outFileLink += fmt::format("_{}", ruhw.flp);
        if (fileFor != "flp") {
          outFileLink += fmt::format("_cru{}_{}", ruhw.cruHWID, link->endPointID);
          if (fileFor != "cruendpoint") {
            outFileLink += fmt::format("_lnk{}_feeid{}", link->idInCRU, link->feeID);
            if (fileFor != "link") {
              throw std::runtime_error("invalid option provided for file grouping");
            }
          }
        }
      }
      outFileLink += ".raw";
      m2r.getWriter().registerLink(link->feeID, link->cruID, link->idInCRU, link->endPointID, outFileLink);
      if (m2r.getVerbosity()) {
        LOG(info) << "RU" << ruID << '(' << ruhw.ruInLayer << " on lr " << ruhw.layer << ") " << link->describe()
                  << " -> " << outFileLink;
      }
    }
  }
}
