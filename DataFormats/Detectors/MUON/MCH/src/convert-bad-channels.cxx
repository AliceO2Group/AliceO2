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

#include <boost/program_options.hpp>
#include <stdexcept>
#include <TFile.h>
#include <string>
#include <iostream>
#include <fmt/format.h>
#include "DataFormatsMCH/DsChannelGroup.h"

namespace po = boost::program_options;

int convertRootToCSV(const std::string rootFileName)
{
  TFile* f = TFile::Open(rootFileName.c_str());
  if (f->IsZombie()) {
    throw std::runtime_error("can not open " + rootFileName);
  }
  auto& tinfo = typeid(std::vector<o2::mch::DsChannelId>*);
  TClass* cl = TClass::GetClass(tinfo);
  auto channels = static_cast<std::vector<o2::mch::DsChannelId>*>(f->GetObjectChecked("ccdb_object", cl));

  std::cout << fmt::format("solarid,dsid,ch\n");

  for (auto c : *channels) {
    std::cout << fmt::format("{},{},{}\n",
                             c.getSolarId(), c.getElinkId(), c.getChannel());
  }

  delete f;
  return 0;
}

int main(int argc, char** argv)
{
  po::variables_map vm;
  po::options_description options;

  // clang-format off
    options.add_options()
     ("help,h","help")
     ("input",po::value<std::string>()->required(),"path to input root file to be converted to csv");
  // clang-format on

  po::options_description cmdline;
  cmdline.add(options);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << R"(
This program converts a Root file containing bad channels information into the
same information in CSV format.

The output file format is :

solarid, dsid, ch

where solarid, dsid and ch are integers.

)";
    std::cout
      << options << "\n";
    std::cout << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    std::cout << options << "\n";
    exit(1);
  }

  return convertRootToCSV(vm["input"].as<std::string>());
}
