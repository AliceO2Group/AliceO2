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
#include "DetectorsRaw/RawFileWriter.h"
#include "DigitEncoder.h"
#include "DigitReader.h"
#include "Framework/Logger.h"
#include "Headers/RAWDataHeader.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHRawEncoderPayload/DataBlock.h"
#include "MCHRawEncoderPayload/PayloadPaginator.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/NameConf.h"

#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>
#include <boost/program_options.hpp>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <filesystem>

namespace po = boost::program_options;
using namespace o2::mch::raw;

namespace
{
std::string asString(const o2::InteractionRecord& ir)
{
  return fmt::format("ORB {:6d} BC {:4d}",
                     ir.orbit, ir.bc);
}

std::string asString(const o2::InteractionTimeRecord& ir)
{
  return asString(static_cast<o2::InteractionRecord>(ir));
}
} // namespace

std::ostream& operator<<(std::ostream& os, const o2::mch::Digit& d)
{
  os << fmt::format("DE {:4d} PADUID {:8d} ADC {:6d} TS {:g}",
                    d.getDetID(), d.getPadID(), d.getADC(),
                    d.getTime());
  return os;
}

void digit2raw(const std::string& input,
               DigitEncoder& encoder,
               PayloadPaginator& paginate)
{
  o2::InteractionRecord ir;
  std::vector<o2::mch::Digit> digits;
  DigitReader dr(input.c_str());

  while (dr.nextDigits(ir, digits)) {

    std::vector<std::byte> buffer;

    encoder(digits, buffer, ir.orbit, ir.bc);

    paginate(buffer);
  }
}

int main(int argc, char* argv[])
{
  po::options_description generic("options");
  bool userLogic{false};
  bool dummyElecMap{false};
  bool chargeSumMode{true};
  std::string input;
  po::variables_map vm;

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("userLogic,u",po::bool_switch(&userLogic),"user logic format")
      ("dummyElecMap,d",po::bool_switch(&dummyElecMap),"use a dummy electronic mapping (for testing only)")
      ("output-dir,o",po::value<std::string>()->default_value("./"),"output directory for file(s)")
      ("input-file,i",po::value<std::string>(&input)->default_value("mchdigits.root"),"input file name")
      ("configKeyValues", po::value<std::string>()->default_value(""), "comma-separated configKeyValues")
      ("no-empty-hbf,e", po::value<bool>()->default_value(true), "do not create empty HBF pages (except for HBF starting TF)")
      ("hbfutils-config", po::value<std::string>()->default_value(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)), "config file for HBFUtils (or none)")
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

  std::string confDig = vm["hbfutils-config"].as<std::string>();
  if (!confDig.empty() && confDig != "none") {
    o2::conf::ConfigurableParam::updateFromFile(confDig, "HBFUtils");
  }
  o2::conf::ConfigurableParam::updateFromString(vm["configKeyValues"].as<std::string>());

  if (vm.count("verbosity")) {
    fair::Logger::SetVerbosity(vm["verbosity"].as<std::string>());
  }

  if (dummyElecMap) {
    std::cout << "WARNING: using dummy electronic mapping\n";
  }

  auto det2elec = (dummyElecMap ? createDet2ElecMapper<ElectronicMapperDummy>() : createDet2ElecMapper<ElectronicMapperGenerated>());

  DigitEncoder encoder = createDigitEncoder(userLogic, det2elec);

  auto solar2feelink = (dummyElecMap ? createSolar2FeeLinkMapper<ElectronicMapperDummy>() : createSolar2FeeLinkMapper<ElectronicMapperGenerated>());

  o2::raw::RawFileWriter fw(o2::header::DAQID(o2::header::DAQID::MCH).getO2Origin());

  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom(inputGRP)};
  fw.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::MCH)); // must be set explicitly

  if (vm["no-empty-hbf"].as<bool>()) {
    fw.setDontFillEmptyHBF(true);
  }

  auto outDirName = vm["output-dir"].as<std::string>();

  // if needed, create output directory
  if (!std::filesystem::exists(outDirName)) {
    if (!std::filesystem::create_directories(outDirName)) {
      LOG(FATAL) << "could not create output directory " << outDirName;
    } else {
      LOG(INFO) << "created output directory " << outDirName;
    }
  }

  fw.writeConfFile("MCH", "RAWDATA", fmt::format("{}/MCHraw.cfg", outDirName));

  std::string output = fmt::format("{}/mch.raw", outDirName);
  PayloadPaginator paginator(fw, output, solar2feelink, userLogic, chargeSumMode);

  digit2raw(input, encoder, paginator);

  o2::raw::HBFUtils::Instance().print();

  return 0;
}
