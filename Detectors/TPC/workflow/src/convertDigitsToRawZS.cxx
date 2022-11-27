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

/// \file convertDigitsToRawZS.cxx
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)

#include <boost/program_options.hpp>

#include <string_view>
#include <memory>
#include <vector>
#include <fmt/format.h>
#include <filesystem>
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"

#include "GPUO2Interface.h"
#include "GPUReconstructionConvert.h"
#include "GPUHostDataTypes.h"
#include "GPUParam.h"

#include "Framework/Logger.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/Helpers.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RDHUtils.h"
#include "TPCBase/RDHUtils.h"
#include "DataFormatsTPC/Digit.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/RDHUtils.h"
#include "TPCReconstruction/IonTailCorrection.h"
#include "GPUO2InterfaceConfiguration.h"

namespace bpo = boost::program_options;

using namespace o2::tpc;
using namespace o2::gpu;
using namespace o2::dataformats;
using o2::MCCompLabel;

constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
constexpr static size_t NEndpoints = o2::gpu::GPUTrackingInOutZS::NENDPOINTS;
using DigitArray = std::array<gsl::span<const o2::tpc::Digit>, Sector::MAXSECTOR>;

static constexpr const char* CRU_FLPS[361] = {
  "alio2-cr1-flp070", "alio2-cr1-flp069", "alio2-cr1-flp070", "alio2-cr1-flp069", "alio2-cr1-flp072", "alio2-cr1-flp071", "alio2-cr1-flp072", "alio2-cr1-flp071", "alio2-cr1-flp072", "alio2-cr1-flp071", "alio2-cr1-flp002", "alio2-cr1-flp001", "alio2-cr1-flp002", "alio2-cr1-flp001", "alio2-cr1-flp004", "alio2-cr1-flp003", "alio2-cr1-flp004", "alio2-cr1-flp003",
  "alio2-cr1-flp004", "alio2-cr1-flp003", "alio2-cr1-flp006", "alio2-cr1-flp005", "alio2-cr1-flp006", "alio2-cr1-flp005", "alio2-cr1-flp008", "alio2-cr1-flp007", "alio2-cr1-flp008", "alio2-cr1-flp007", "alio2-cr1-flp008", "alio2-cr1-flp007", "alio2-cr1-flp010", "alio2-cr1-flp009", "alio2-cr1-flp010", "alio2-cr1-flp009", "alio2-cr1-flp012", "alio2-cr1-flp011",
  "alio2-cr1-flp012", "alio2-cr1-flp011", "alio2-cr1-flp012", "alio2-cr1-flp011", "alio2-cr1-flp014", "alio2-cr1-flp013", "alio2-cr1-flp014", "alio2-cr1-flp013", "alio2-cr1-flp016", "alio2-cr1-flp015", "alio2-cr1-flp016", "alio2-cr1-flp015", "alio2-cr1-flp016", "alio2-cr1-flp015", "alio2-cr1-flp018", "alio2-cr1-flp017", "alio2-cr1-flp018", "alio2-cr1-flp017",
  "alio2-cr1-flp020", "alio2-cr1-flp019", "alio2-cr1-flp020", "alio2-cr1-flp019", "alio2-cr1-flp020", "alio2-cr1-flp019", "alio2-cr1-flp022", "alio2-cr1-flp021", "alio2-cr1-flp022", "alio2-cr1-flp021", "alio2-cr1-flp024", "alio2-cr1-flp023", "alio2-cr1-flp024", "alio2-cr1-flp023", "alio2-cr1-flp024", "alio2-cr1-flp023", "alio2-cr1-flp026", "alio2-cr1-flp025",
  "alio2-cr1-flp026", "alio2-cr1-flp025", "alio2-cr1-flp028", "alio2-cr1-flp027", "alio2-cr1-flp028", "alio2-cr1-flp027", "alio2-cr1-flp028", "alio2-cr1-flp027", "alio2-cr1-flp030", "alio2-cr1-flp029", "alio2-cr1-flp030", "alio2-cr1-flp029", "alio2-cr1-flp032", "alio2-cr1-flp031", "alio2-cr1-flp032", "alio2-cr1-flp031", "alio2-cr1-flp032", "alio2-cr1-flp031",
  "alio2-cr1-flp034", "alio2-cr1-flp033", "alio2-cr1-flp034", "alio2-cr1-flp033", "alio2-cr1-flp036", "alio2-cr1-flp035", "alio2-cr1-flp036", "alio2-cr1-flp035", "alio2-cr1-flp036", "alio2-cr1-flp035", "alio2-cr1-flp038", "alio2-cr1-flp037", "alio2-cr1-flp038", "alio2-cr1-flp037", "alio2-cr1-flp040", "alio2-cr1-flp039", "alio2-cr1-flp040", "alio2-cr1-flp039",
  "alio2-cr1-flp040", "alio2-cr1-flp039", "alio2-cr1-flp042", "alio2-cr1-flp041", "alio2-cr1-flp042", "alio2-cr1-flp041", "alio2-cr1-flp044", "alio2-cr1-flp043", "alio2-cr1-flp044", "alio2-cr1-flp043", "alio2-cr1-flp044", "alio2-cr1-flp043", "alio2-cr1-flp046", "alio2-cr1-flp045", "alio2-cr1-flp046", "alio2-cr1-flp045", "alio2-cr1-flp048", "alio2-cr1-flp047",
  "alio2-cr1-flp048", "alio2-cr1-flp047", "alio2-cr1-flp048", "alio2-cr1-flp047", "alio2-cr1-flp050", "alio2-cr1-flp049", "alio2-cr1-flp050", "alio2-cr1-flp049", "alio2-cr1-flp052", "alio2-cr1-flp051", "alio2-cr1-flp052", "alio2-cr1-flp051", "alio2-cr1-flp052", "alio2-cr1-flp051", "alio2-cr1-flp054", "alio2-cr1-flp053", "alio2-cr1-flp054", "alio2-cr1-flp053",
  "alio2-cr1-flp056", "alio2-cr1-flp055", "alio2-cr1-flp056", "alio2-cr1-flp055", "alio2-cr1-flp056", "alio2-cr1-flp055", "alio2-cr1-flp058", "alio2-cr1-flp057", "alio2-cr1-flp058", "alio2-cr1-flp057", "alio2-cr1-flp060", "alio2-cr1-flp059", "alio2-cr1-flp060", "alio2-cr1-flp059", "alio2-cr1-flp060", "alio2-cr1-flp059", "alio2-cr1-flp062", "alio2-cr1-flp061",
  "alio2-cr1-flp062", "alio2-cr1-flp061", "alio2-cr1-flp064", "alio2-cr1-flp063", "alio2-cr1-flp064", "alio2-cr1-flp063", "alio2-cr1-flp064", "alio2-cr1-flp063", "alio2-cr1-flp066", "alio2-cr1-flp065", "alio2-cr1-flp066", "alio2-cr1-flp065", "alio2-cr1-flp068", "alio2-cr1-flp067", "alio2-cr1-flp068", "alio2-cr1-flp067", "alio2-cr1-flp068", "alio2-cr1-flp067",
  "alio2-cr1-flp074", "alio2-cr1-flp073", "alio2-cr1-flp074", "alio2-cr1-flp073", "alio2-cr1-flp076", "alio2-cr1-flp075", "alio2-cr1-flp076", "alio2-cr1-flp075", "alio2-cr1-flp076", "alio2-cr1-flp075", "alio2-cr1-flp078", "alio2-cr1-flp077", "alio2-cr1-flp078", "alio2-cr1-flp077", "alio2-cr1-flp080", "alio2-cr1-flp079", "alio2-cr1-flp080", "alio2-cr1-flp079",
  "alio2-cr1-flp080", "alio2-cr1-flp079", "alio2-cr1-flp082", "alio2-cr1-flp081", "alio2-cr1-flp082", "alio2-cr1-flp081", "alio2-cr1-flp084", "alio2-cr1-flp083", "alio2-cr1-flp084", "alio2-cr1-flp083", "alio2-cr1-flp084", "alio2-cr1-flp083", "alio2-cr1-flp086", "alio2-cr1-flp085", "alio2-cr1-flp086", "alio2-cr1-flp085", "alio2-cr1-flp088", "alio2-cr1-flp087",
  "alio2-cr1-flp088", "alio2-cr1-flp087", "alio2-cr1-flp088", "alio2-cr1-flp087", "alio2-cr1-flp090", "alio2-cr1-flp089", "alio2-cr1-flp090", "alio2-cr1-flp089", "alio2-cr1-flp092", "alio2-cr1-flp091", "alio2-cr1-flp092", "alio2-cr1-flp091", "alio2-cr1-flp092", "alio2-cr1-flp091", "alio2-cr1-flp094", "alio2-cr1-flp093", "alio2-cr1-flp094", "alio2-cr1-flp093",
  "alio2-cr1-flp096", "alio2-cr1-flp095", "alio2-cr1-flp096", "alio2-cr1-flp095", "alio2-cr1-flp096", "alio2-cr1-flp095", "alio2-cr1-flp098", "alio2-cr1-flp097", "alio2-cr1-flp098", "alio2-cr1-flp097", "alio2-cr1-flp100", "alio2-cr1-flp099", "alio2-cr1-flp100", "alio2-cr1-flp099", "alio2-cr1-flp100", "alio2-cr1-flp099", "alio2-cr1-flp102", "alio2-cr1-flp101",
  "alio2-cr1-flp102", "alio2-cr1-flp101", "alio2-cr1-flp104", "alio2-cr1-flp103", "alio2-cr1-flp104", "alio2-cr1-flp103", "alio2-cr1-flp104", "alio2-cr1-flp103", "alio2-cr1-flp106", "alio2-cr1-flp105", "alio2-cr1-flp106", "alio2-cr1-flp105", "alio2-cr1-flp108", "alio2-cr1-flp107", "alio2-cr1-flp108", "alio2-cr1-flp107", "alio2-cr1-flp108", "alio2-cr1-flp107",
  "alio2-cr1-flp110", "alio2-cr1-flp109", "alio2-cr1-flp110", "alio2-cr1-flp109", "alio2-cr1-flp112", "alio2-cr1-flp111", "alio2-cr1-flp112", "alio2-cr1-flp111", "alio2-cr1-flp112", "alio2-cr1-flp111", "alio2-cr1-flp114", "alio2-cr1-flp113", "alio2-cr1-flp114", "alio2-cr1-flp113", "alio2-cr1-flp116", "alio2-cr1-flp115", "alio2-cr1-flp116", "alio2-cr1-flp115",
  "alio2-cr1-flp116", "alio2-cr1-flp115", "alio2-cr1-flp118", "alio2-cr1-flp117", "alio2-cr1-flp118", "alio2-cr1-flp117", "alio2-cr1-flp120", "alio2-cr1-flp119", "alio2-cr1-flp120", "alio2-cr1-flp119", "alio2-cr1-flp120", "alio2-cr1-flp119", "alio2-cr1-flp122", "alio2-cr1-flp121", "alio2-cr1-flp122", "alio2-cr1-flp121", "alio2-cr1-flp124", "alio2-cr1-flp123",
  "alio2-cr1-flp124", "alio2-cr1-flp123", "alio2-cr1-flp124", "alio2-cr1-flp123", "alio2-cr1-flp126", "alio2-cr1-flp125", "alio2-cr1-flp126", "alio2-cr1-flp125", "alio2-cr1-flp128", "alio2-cr1-flp127", "alio2-cr1-flp128", "alio2-cr1-flp127", "alio2-cr1-flp128", "alio2-cr1-flp127", "alio2-cr1-flp130", "alio2-cr1-flp129", "alio2-cr1-flp130", "alio2-cr1-flp129",
  "alio2-cr1-flp132", "alio2-cr1-flp131", "alio2-cr1-flp132", "alio2-cr1-flp131", "alio2-cr1-flp132", "alio2-cr1-flp131", "alio2-cr1-flp134", "alio2-cr1-flp133", "alio2-cr1-flp134", "alio2-cr1-flp133", "alio2-cr1-flp136", "alio2-cr1-flp135", "alio2-cr1-flp136", "alio2-cr1-flp135", "alio2-cr1-flp136", "alio2-cr1-flp135", "alio2-cr1-flp138", "alio2-cr1-flp137",
  "alio2-cr1-flp138", "alio2-cr1-flp137", "alio2-cr1-flp140", "alio2-cr1-flp139", "alio2-cr1-flp140", "alio2-cr1-flp139", "alio2-cr1-flp140", "alio2-cr1-flp139", "alio2-cr1-flp142", "alio2-cr1-flp141", "alio2-cr1-flp142", "alio2-cr1-flp141", "alio2-cr1-flp144", "alio2-cr1-flp143", "alio2-cr1-flp144", "alio2-cr1-flp143", "alio2-cr1-flp144", "alio2-cr1-flp143",
  "alio2-cr1-flp145"};

struct ProcessAttributes {
  std::unique_ptr<unsigned long long int[]> zsoutput;
  std::vector<unsigned int> sizes;
  std::function<void(std::vector<o2::tpc::Digit>&)> digitsFilter = nullptr;
  MCLabelContainer mctruthArray;
  std::vector<int> inputIds;
  int version = 2;
  float zsThreshold = 2.f;
  bool padding = true;
  int verbosity = 1;
};

void convert(DigitArray& inputDigits, ProcessAttributes* processAttributes, o2::raw::RawFileWriter& writer);
#include "DetectorsRaw/HBFUtils.h"
void convertDigitsToZSfinal(std::string_view digitsFile, std::string_view outputPath, std::string_view fileFor,
                            bool sectorBySector, uint32_t rdhV, uint32_t zsV, bool stopPage, bool padding, bool createParentDir)
{
  // ===| open file and get tree |==============================================
  std::unique_ptr<TFile> o2simDigits(TFile::Open(digitsFile.data()));
  if (!o2simDigits || !o2simDigits->IsOpen() || o2simDigits->IsZombie()) {
    LOGP(error, "Could not open file {}", digitsFile.data());
    exit(1);
  }
  auto treeSim = (TTree*)o2simDigits->Get("o2sim");
  if (!treeSim) {
    LOGP(error, "Could not read digits tree from file {}", digitsFile.data());
    exit(1);
  }

  gROOT->cd();

  // ===| set up output directory |=============================================
  std::string outDir{outputPath};
  if (outDir.empty()) {
    outDir = "./";
  }
  if (outDir.back() != '/') {
    outDir += '/';
  }

  // if needed, create output directory
  if (!std::filesystem::exists(outDir)) {
    if (createParentDir) {
      o2::raw::assertOutputDirectory(outDir);
    } else {
      LOGP(error, "Requested output directory '{}' does not exists, consider removing '-n'", outDir);
      exit(1);
    }
  }

  // ===| set up raw writer |===================================================
  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);

  o2::raw::RawFileWriter writer{"TPC"}; // to set the RDHv6.sourceID if V6 is used
  writer.useRDHVersion(rdhV);
  writer.setAddSeparateHBFStopPage(stopPage);
  writer.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::TPC)); // must be set explicitly

  for (unsigned int i = 0; i < NSectors + 1; i++) {
    for (unsigned int j = 0; j < NEndpoints; j++) {
      const unsigned int cruInSector = j / 2;
      const unsigned int cruID = i * 10 + cruInSector;
      const unsigned int defaultLink = i == NSectors ? rdh_utils::SACLinkID : zsV <= 2 ? rdh_utils::UserLogicLinkID : zsV == 3 ? rdh_utils::ILBZSLinkID : rdh_utils::DLBZSLinkID;
      const rdh_utils::FEEIDType feeid = i == NSectors ? 46208 : rdh_utils::getFEEID(cruID, j & 1, defaultLink);
      std::string outfname;
      if (fileFor == "all") { // single file for all links
        outfname = fmt::format("{}tpc_all.raw", outDir);
      } else if (fileFor == "sector") {
        outfname = i == NSectors ? fmt::format("{}tpc_iac.raw", outDir) : fmt::format("{}tpc_sector{}.raw", outDir, i);
      } else if (fileFor == "link" || fileFor == "cruendpoint") {
        outfname = fmt::format("{}TPC_{}_cru{}_{}.raw", outDir, CRU_FLPS[cruID], cruID, j & 1);
      } else {
        throw std::runtime_error("invalid option provided for file grouping");
      }
      writer.registerLink(feeid, cruID, defaultLink, j & 1, outfname);
      if (i == NSectors) {
        break; // Special IAC node
      }
    }
  }
  if (fileFor != "link") { // in case >1 link goes to the file, we must cache to preserve the TFs ordering
    writer.useCaching();
  }
  writer.doLazinessCheck(false); // LazinessCheck is not thread-safe

  // ===| set up branch addresses |=============================================
  std::vector<Digit>* vDigitsPerSectorCollection[Sector::MAXSECTOR] = {nullptr}; // container that keeps Digits per sector

  treeSim->SetBranchStatus("*", 0);
  treeSim->SetBranchStatus("TPCDigit_*", 1);

  ProcessAttributes attr;
  attr.padding = padding;
  attr.version = zsV;

  GPUO2InterfaceConfiguration config;
  auto globalConfig = config.ReadConfigurableParam();
  attr.zsThreshold = config.configReconstruction.tpc.zsThreshold;
  if (globalConfig.zsOnTheFlyDigitsFilter) {
    attr.digitsFilter = [](std::vector<o2::tpc::Digit>& digits) {
      IonTailCorrection itCorr;
      itCorr.filterDigitsDirect(digits);
    };
  }

  for (int iSecBySec = 0; iSecBySec < Sector::MAXSECTOR; ++iSecBySec) {
    treeSim->ResetBranchAddresses();
    for (int iSec = 0; iSec < Sector::MAXSECTOR; ++iSec) {
      if (sectorBySector) {
        iSec = iSecBySec;
      }
      vDigitsPerSectorCollection[iSec] = nullptr;
      treeSim->SetBranchAddress(TString::Format("TPCDigit_%d", iSec), &vDigitsPerSectorCollection[iSec]);
      if (sectorBySector) {
        break;
      }
    }
    for (Long64_t ievent = 0; ievent < treeSim->GetEntries(); ++ievent) {
      DigitArray inputDigits;
      if (sectorBySector) {
        treeSim->GetBranch(TString::Format("TPCDigit_%d", iSecBySec))->GetEntry(ievent);
      } else {
        treeSim->GetEntry(ievent);
      }

      for (int iSec = 0; iSec < Sector::MAXSECTOR; ++iSec) {
        if (sectorBySector) {
          iSec = iSecBySec;
        }
        inputDigits[iSec] = *vDigitsPerSectorCollection[iSec]; //????
        if (sectorBySector) {
          break;
        }
      }
      convert(inputDigits, &attr, writer);
      for (int iSec = 0; iSec < Sector::MAXSECTOR; ++iSec) {
        delete vDigitsPerSectorCollection[iSec];
        vDigitsPerSectorCollection[iSec] = nullptr;
      }
    }
    if (!sectorBySector) {
      break;
    }
  }
  // for further use we write the configuration file for the output
  writer.writeConfFile("TPC", "RAWDATA", fmt::format("{}tpcraw.cfg", outDir));
}

void convert(DigitArray& inputDigits, ProcessAttributes* processAttributes, o2::raw::RawFileWriter& writer)
{
  const auto zsThreshold = processAttributes->zsThreshold;
  GPUParam _GPUParam;
  _GPUParam.SetDefaults(5.00668);
  const GPUParam mGPUParam = _GPUParam;

  o2::InteractionRecord ir = o2::raw::HBFUtils::Instance().getFirstSampledTFIR();
  ir.bc = 0; // By convention the TF starts at BC = 0
  o2::gpu::GPUReconstructionConvert::RunZSEncoder(inputDigits, nullptr, nullptr, &writer, &ir, mGPUParam, processAttributes->version, false, zsThreshold, processAttributes->padding, processAttributes->digitsFilter);
}

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       " <cmds/options>\n"
                                       "  Tool will convert simulation digits to raw zero suppressed data\n"
                                       "Commands / Options");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbose,v", bpo::value<uint32_t>()->default_value(0), "Select verbosity level [0 = no output]");
    add_option("input-file,i", bpo::value<std::string>()->default_value("tpcdigits.root"), "Specifies input file.");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "Specify output directory");
    add_option("no-parent-directories,n", "Do not create parent directories recursively");
    add_option("sector-by-sector,s", bpo::value<bool>()->default_value(false)->implicit_value(true), "Run one TPC sector after another");
    add_option("file-for,f", bpo::value<std::string>()->default_value("sector"), "single file per: link,sector,cruendpoint,all");
    add_option("stop-page,p", bpo::value<bool>()->default_value(false)->implicit_value(true), "HBF stop on separate CRU page");
    add_option("padding", bpo::value<bool>()->default_value(false)->implicit_value(true), "Pad all pages to 8kb");
    uint32_t defRDH = o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>();
    add_option("hbfutils-config,u", bpo::value<std::string>()->default_value(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)), "config file for HBFUtils (or none)");
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(defRDH), "RDH version to use");
    add_option("zs-version,r", bpo::value<uint32_t>()->default_value(2), "ZS version to use");
    add_option("configKeyValues", bpo::value<std::string>()->default_value(""), "comma-separated configKeyValues");

    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help") || argc == 1) {
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
  convertDigitsToZSfinal(
    vm["input-file"].as<std::string>(),
    vm["output-dir"].as<std::string>(),
    vm["file-for"].as<std::string>(),
    vm["sector-by-sector"].as<bool>(),
    vm["rdh-version"].as<uint32_t>(),
    vm["zs-version"].as<uint32_t>(),
    vm["stop-page"].as<bool>(),
    vm["padding"].as<bool>(),
    !vm.count("no-parent-directories"));

  o2::raw::HBFUtils::Instance().print();

  return 0;
}
