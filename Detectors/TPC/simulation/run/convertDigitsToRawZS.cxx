// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TPCBase/Digit.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "DataFormatsTPC/Helpers.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RDHUtils.h"
#include "TPCBase/RDHUtils.h"
#include "TPCBase/Digit.h"
#include "CommonUtils/ConfigurableParam.h"

namespace bpo = boost::program_options;

using namespace o2::tpc;
using namespace o2::gpu;
using o2::MCCompLabel;

constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;
constexpr static size_t NEndpoints = o2::gpu::GPUTrackingInOutZS::NENDPOINTS;
using DigitArray = std::array<gsl::span<const o2::tpc::Digit>, Sector::MAXSECTOR>;
using MCLabelContainer = o2::dataformats::MCTruthContainer<MCCompLabel>;

struct ProcessAttributes {
  std::unique_ptr<unsigned long long int[]> zsoutput;
  std::vector<unsigned int> sizes;
  MCLabelContainer mctruthArray;
  std::unique_ptr<o2::gpu::GPUReconstructionConvert> zsEncoder;
  std::vector<int> inputIds;
  bool zs12bit = true;
  float zsThreshold = 2.f;
  int verbosity = 1;
};

void convert(DigitArray& inputDigits, ProcessAttributes* processAttributes, o2::raw::RawFileWriter& writer);
#include "DetectorsRaw/HBFUtils.h"
void convertDigitsToZSfinal(std::string_view digitsFile, std::string_view outputPath, bool sectorBySector, uint32_t rdhV)
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

  // ===| set up branch addresses |=============================================
  std::vector<Digit>* vDigitsPerSectorCollection[Sector::MAXSECTOR] = {nullptr}; // container that keeps Digits per sector

  ProcessAttributes attr;

  // raw data output
  o2::raw::RawFileWriter writer{"TPC"}; // to set the RDHv6.sourceID if V6 is used
  writer.useRDHVersion(rdhV);

  const unsigned int defaultLink = rdh_utils::UserLogicLinkID;

  // set up raw writer
  std::string outDir{outputPath};
  if (outDir.empty()) {
    outDir = "./";
  }
  if (outDir.back() != '/') {
    outDir += '/';
  }
  for (unsigned int i = 0; i < NSectors; i++) {
    for (unsigned int j = 0; j < NEndpoints; j++) {
      const unsigned int cruInSector = j / 2;
      const unsigned int cruID = i * 10 + cruInSector;
      const rdh_utils::FEEIDType feeid = rdh_utils::getFEEID(cruID, j & 1, defaultLink);
      writer.registerLink(feeid, cruID, defaultLink, j & 1, fmt::format("{}cru{}_{}.raw", outDir, cruID, j & 1));
    }
  }

  treeSim->SetBranchStatus("*", 0);
  treeSim->SetBranchStatus("TPCDigit_*", 1);
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
  auto& zsEncoder = processAttributes->zsEncoder;
  const auto zsThreshold = processAttributes->zsThreshold;
  const auto zs12bit = processAttributes->zs12bit;
  GPUParam _GPUParam;
  _GPUParam.SetDefaults(5.00668);
  const GPUParam mGPUParam = _GPUParam;

  o2::InteractionRecord ir = o2::raw::HBFUtils::Instance().getFirstIR();
  zsEncoder->RunZSEncoder<o2::tpc::Digit>(inputDigits, nullptr, nullptr, &writer, &ir, mGPUParam, zs12bit, false, zsThreshold);
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
    add_option("input-file,i", bpo::value<std::string>()->required(), "Specifies input file.");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "Specify output directory");
    add_option("sector-by-sector,s", bpo::value<bool>()->default_value(false), "Run one TPC sector after another");
    uint32_t defRDH = o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>();
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(defRDH), "RDH version to use");
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
  o2::conf::ConfigurableParam::updateFromString(vm["configKeyValues"].as<std::string>());
  convertDigitsToZSfinal(
    vm["input-file"].as<std::string>(),
    vm["output-dir"].as<std::string>(),
    vm["sector-by-sector"].as<bool>(),
    vm["rdh-version"].as<uint32_t>());

  return 0;
}
