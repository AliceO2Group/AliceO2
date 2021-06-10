// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsMCH/Digit.h"
#include "MCHRawEncoderDigit/Digit2ElecMapper.h"
#include "DigitTreeReader.h"
#include "MCHRawElecMap/Mapper.h"
#include <TFile.h>
#include <boost/program_options.hpp>
#include <fmt/format.h>
#include <iostream>
#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <string>

namespace po = boost::program_options;
using namespace o2::mch::raw;

std::string digitIdAsString(const o2::mch::Digit& digit,
                            const Digit2ElecMapper& digit2elec)
{
  auto optElecId = digit2elec(digit);
  if (!optElecId.has_value()) {
    return "UNKNOWN";
  }
  auto dsElecId = optElecId.value().first;
  auto dschid = optElecId.value().second;
  return fmt::format("{}-CH{}", asString(dsElecId), dschid);
}

void outputToJson(const std::vector<o2::mch::Digit>& digits,
                  std::function<std::optional<DsElecId>(DsDetId)> det2elec,
                  rapidjson::Writer<rapidjson::OStreamWrapper>& writer)

{

  auto digit2elec = createDigit2ElecMapper(det2elec);

  writer.StartArray();
  for (auto d : digits) {
    auto sid = digitIdAsString(d, digit2elec);
    if (sid == "UNKNOWN") {
      continue;
    }
    writer.StartObject();
    writer.Key("id");
    writer.String(sid.c_str());
    writer.Key("adc");
    writer.Int(d.getADC());
    writer.EndObject();
  }
  writer.EndArray();
}

int main(int argc, char* argv[])
{
  po::options_description generic("options");
  bool dummyElecMap{false};
  std::string input;
  po::variables_map vm;

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("dummyElecMap,d",po::bool_switch(&dummyElecMap),"use a dummy electronic mapping (for testing only)")
      ("infile,i",po::value<std::string>(&input)->required(),"input file name");
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

  auto det2elec = (dummyElecMap ? createDet2ElecMapper<ElectronicMapperDummy>() : createDet2ElecMapper<ElectronicMapperGenerated>());

  rapidjson::OStreamWrapper osw(std::cout);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);

  TFile fin(input.c_str());
  if (!fin.IsOpen()) {
    return 3;
  }
  TTree* tree = static_cast<TTree*>(fin.Get("o2sim"));
  if (!tree) {
    return 4;
  }
  o2::mch::ROFRecord rof;
  std::vector<o2::mch::Digit> digits;
  DigitTreeReader dr(tree);

  while (dr.nextDigits(rof, digits)) {
    writer.StartObject();
    writer.Key("orbit");
    writer.Int(rof.getBCData().orbit);
    writer.Key("bc");
    writer.Int(rof.getBCData().bc);
    writer.Key("digits");
    outputToJson(digits, det2elec, writer);
    writer.EndObject();
  }
  return 0;
}
