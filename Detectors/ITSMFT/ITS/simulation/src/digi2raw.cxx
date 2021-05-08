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
#include "DetectorsCommonDataFormats/NameConf.h"
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
    add_option("file-for,f", bpo::value<std::string>()->default_value("layer"), "single file per: all,layer,cru,link");
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
  LOG(INFO) << "HBFUtils settings used for conversion:";

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

  LOG(INFO) << "HBFUtil settings:";
  o2::raw::HBFUtils::Instance().print();

  // if needed, create output directory
  if (!std::filesystem::exists(outDir)) {
    if (!std::filesystem::create_directories(outDir)) {
      LOG(FATAL) << "could not create output directory " << outDir;
    } else {
      LOG(INFO) << "created output directory " << outDir;
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
    LOG(FATAL) << "Failed to find the branch " << digBranchName << " in the tree " << digTreeName;
  }
  digTree.SetBranchAddress(digBranchName.c_str(), &digiVecP);

  // ROF record entries in the digit tree
  ROFRVEC rofRecVec, *rofRecVecP = &rofRecVec;
  std::string rofRecName = o2::utils::Str::concat_string(MAP::getName(), "DigitROF");
  if (!digTree.GetBranch(rofRecName.c_str())) {
    LOG(FATAL) << "Failed to find the branch " << rofRecName << " in the tree " << digTreeName;
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
        LOG(INFO) << "Processing ROF:" << rofRec.getROFrame() << " with " << nDigROF << " entries";
        rofRec.print();
      }
      if (!nDigROF) {
        if (verbosity) {
          LOG(INFO) << "Frame is empty"; // ??
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
  //------------------------------------------------------------------------------->>>>
  // just as an example, we require here that the staves are read via 3 links, with partitioning according to lnkXB below
  // while OB staves use only 1 link.
  // Note, that if the RU container is not defined, it will be created automatically
  // during encoding.
  // If the links of the container are not defined, a single link readout will be assigned

  constexpr int MaxLinksPerRU = 3;
  constexpr int MaxLinksPerCRU = 16;
  const auto& mp = m2r.getMapping();
  int lnkAssign[3][MaxLinksPerRU] = {
    // requested link cabling for IB, MB and OB
    /* // uncomment this to have 1 link per RU
    {9, 0, 0}, // IB
    {16, 0, 0}, // MB
    {28, 0, 0} // OB
     */
    /* //  uncomment this to have 3 link per RU
    {3, 3, 3}, // IB
    {5, 5, 6}, // MB
    {9, 9, 10} // OB
    */
    // partition according to https://indico.cern.ch/event/907685/contributions/3818967/attachments/2019436/3376161/20200411_SOX_EOX_Proposal.pdf
    {3, 3, 3},  // IB
    {8, 8, 0},  // MB
    {14, 14, 0} // OB
  };
  std::vector<std::vector<int>> defRU{// number of RUs per CRU at each layer
                                      {6, 6},
                                      {8, 8},
                                      {5, 5, 5, 5},
                                      {12, 12},
                                      {8, 7, 8, 7},
                                      {11, 10, 11, 10},
                                      {12, 12, 12, 12}};
  // this is an arbitrary mapping
  int nCRU = 0, nRUtot = 0, nRU = 0, nLinks = 0;
  int cruID = 0;
  std::string outFileLink;

  for (int ilr = 0; ilr < mp.NLayers; ilr++) {
    int nruLr = mp.getNStavesOnLr(ilr);
    int ruType = mp.getRUType(nRUtot); // IB, MB or OB
    int* lnkAs = lnkAssign[ruType];
    // count requested number of links per RU
    int nlk = 0;
    for (int i = 3; i--;) {
      nlk += lnkAs[i] ? 1 : 0;
    }
    const auto& ruList = defRU[ilr];
    // sanity check
    if (std::accumulate(ruList.begin(), ruList.end(), 0) != nruLr) {
      throw std::runtime_error("Declared RU partitioning does not add up to expected N staves per layer");
    }
    int ruInLr = 0; // counter for RUs within the layer
    int nCRUlr = ruList.size();
    for (int iCRU = 0; iCRU < nCRUlr; iCRU++) {
      cruID = nCRU * 100 + o2::detectors::DetID::ITS;
      int linkID = 0;
      for (int irc = 0; irc < ruList[iCRU]; irc++) {
        int ruID = nRUtot++;
        bool acceptRU = !(ruID < m2r.getRUSWMin() || ruID > m2r.getRUSWMax()); // ignored RUs ?
        if (acceptRU) {
          m2r.getCreateRUDecode(ruID); // create RU container
          nRU++;
        }
        int accL = 0;
        for (int il = 0; il < MaxLinksPerRU; il++) { // create links
          if (acceptRU && lnkAs[il]) {
            nLinks++;
            auto& ru = *m2r.getRUDecode(ruID);
            uint32_t lanes = mp.getCablesOnRUType(ru.ruInfo->ruType); // lanes pattern of this RU
            ru.links[il] = m2r.addGBTLink();
            auto link = m2r.getGBTLink(ru.links[il]);
            link->lanes = lanes & (((0x1 << lnkAs[il]) - 1) << (accL)); // RS FIXME
            link->idInCRU = linkID;
            link->cruID = cruID;
            link->feeID = mp.RUSW2FEEId(ruID, il);
            link->endPointID = 0; // 0 or 1
            accL += lnkAs[il];
            if (m2r.getVerbosity()) {
              LOG(INFO) << "RU" << ruID << '(' << ruInLr << " on lr " << ilr << ") " << link->describe()
                        << " -> " << outFileLink;
            }
            // register the link in the writer, if not done here, its data will be dumped to common default file

            if (fileFor == "all") { // single file for all links
              outFileLink = o2::utils::Str::concat_string(outDir, "/", outPrefix, ".raw");
            } else if (fileFor == "layer") {
              outFileLink = o2::utils::Str::concat_string(outDir, "/", outPrefix, "_lr", std::to_string(ilr), ".raw");
            } else if (fileFor == "cru") {
              outFileLink = o2::utils::Str::concat_string(outDir, "/", outPrefix, "_cru", std::to_string(cruID), ".raw");
            } else if (fileFor == "link") {
              outFileLink = o2::utils::Str::concat_string(outDir, "/", outPrefix, "_cru", std::to_string(cruID),
                                                          "_link", std::to_string(linkID), "_ep", std::to_string(link->endPointID),
                                                          "_feeid", std::to_string(link->feeID), ".raw");
            } else {
              throw std::runtime_error("invalid option provided for file grouping");
            }
            m2r.getWriter().registerLink(link->feeID, link->cruID, link->idInCRU, link->endPointID, outFileLink);
            //
            linkID++;
          }
        } // links of RU
        ruInLr++;
      } // RUs of CRU
      nCRU++;
    } // CRUs of the layer
  }   // layers
  LOG(INFO) << "Distributed " << nLinks << " links on " << nRU << " RUs in " << nCRU << " CRUs";
}
