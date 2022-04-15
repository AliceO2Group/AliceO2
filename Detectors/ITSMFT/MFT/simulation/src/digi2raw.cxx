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
/// \author ruben.shahoyan@cern.ch, bogdan.vulpescu@clermont.in2p3.fr

#include <boost/program_options.hpp>
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include <TStopwatch.h>
#include "Framework/Logger.h"
#include <vector>
#include <string>
#include <iomanip>
#include <filesystem>
#include "ITSMFTReconstruction/ChipMappingMFT.h"
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

/// MC->raw conversion with new (variable page size) format for MFT
using MAP = o2::itsmft::ChipMappingMFT;
namespace bpo = boost::program_options;

void setupLinks(o2::itsmft::MC2RawEncoder<MAP>& m2r, std::string_view outDir, std::string_view outPrefix, std::string_view fileFor);
void digi2raw(std::string_view inpName, std::string_view outDir, std::string_view fileFor, int verbosity, uint32_t rdhV = 4, bool noEmptyHBF = false,
              int superPageSizeInB = 1024 * 1024);

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Convert MFT digits to CRU raw data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbosity,v", bpo::value<uint32_t>()->default_value(0), "verbosity level [0 = no output]");
    add_option("input-file,i", bpo::value<std::string>()->default_value("mftdigits.root"), "input  digits file");
    add_option("file-for,f", bpo::value<std::string>()->default_value("layer"), "single file per: all,layer,cru,link");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    uint32_t defRDH = o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>();
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(defRDH), "RDH version to use");
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
  int lastTreeID = -1;
  long offs = 0, nEntProc = 0;
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

void setupLinks(o2::itsmft::MC2RawEncoder<MAP>& m2r, std::string_view outDir, std::string_view outPrefix, std::string_view fileFor)
{
  // see the same file from ITS
  const auto& mp = m2r.getMapping();
  std::string outFileLink;
  o2::itsmft::ChipMappingMFT mftMapping;
  auto& ruMapping = mftMapping.getRUMappingData();

  for (UInt_t ruID = 0; ruID < mftMapping.getNRUs(); ruID++) {

    if (ruID < m2r.getRUSWMin() || ruID > m2r.getRUSWMax()) { // ignored RUs ?
      continue;
    }

    auto linkID = ruMapping[ruID].idInCRU;
    auto feeID = mftMapping.RUSW2FEEId(ruID, linkID);
    uint16_t layer, ruOnLayer, zone = feeID & 0x3;
    mftMapping.expandFEEId(feeID, layer, ruOnLayer, linkID);

    m2r.getCreateRUDecode(ruID); // create RU container

    auto& ru = *m2r.getRUDecode(ruID);
    ru.links[0] = m2r.addGBTLink();
    uint32_t lanes = mp.getCablesOnRUType(mp.getRUType(zone, layer)); // lanes pattern of this RU
    auto link = m2r.getGBTLink(ru.links[0]);
    link->lanes = lanes;
    link->feeID = feeID;
    link->idInCRU = linkID;
    link->cruID = ruMapping[ruID].cruHWID;       // CRU Serial Number
    link->endPointID = ruMapping[ruID].endpoint; // endpoint = face
    outFileLink = o2::utils::Str::concat_string(outDir, "/", outPrefix);
    if (fileFor != "all") { // single file for all links
      outFileLink += fmt::format("_{}", ruMapping[ruID].flp);
      if (fileFor != "flp") {
        outFileLink += fmt::format("_cru{}_{}", ruMapping[ruID].cruHWID, link->endPointID);
        if (fileFor != "cru") {
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
      LOG(info) << "RU" << ruID << '(' << ruMapping[ruID].cruHWID << " on idInCRU " << ruMapping[ruID].idInCRU << ") " << link->describe()
                << " -> " << outFileLink;
    }
  }
}
