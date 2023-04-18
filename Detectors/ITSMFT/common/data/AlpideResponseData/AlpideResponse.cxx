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
///

#include <boost/program_options.hpp>
#include "ITSMFTSimulation/AlpideSimResponse.h"
#include <TFile.h>
#include <TSystem.h>
#include <cstdio>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

void alpideResponse(const std::string& inpath = "./",
                    const std::string& outpath = "./",
                    const std::string& response_file = "AlpideResponseData.root")
{

  o2::itsmft::AlpideSimResponse resp0, resp1;

  resp0.initData(0, inpath.data());
  resp1.initData(1, inpath.data());

  auto file = TFile::Open((outpath + response_file).data(), "recreate");
  file->WriteObjectAny(&resp0, "o2::itsmft::AlpideSimResponse", "response0");
  file->WriteObjectAny(&resp1, "o2::itsmft::AlpideSimResponse", "response1");
  file->Close();
}

int main(int argc, const char* argv[])
{
  namespace bpo = boost::program_options;
  bpo::variables_map vm;
  bpo::options_description options("Alpide reponse generator options");
  options.add_options()(
    "inputdir,i", bpo::value<std::string>()->default_value("./"), "Path where Vbb-0.0V and Vbb-3.0V are located.")(
    "outputdir,o", bpo::value<std::string>()->default_value("./"), "Path where to store the output.")(
    "name,n", bpo::value<std::string>()->default_value("AlpideResponseData.root"), "Output file name.");

  try {
    bpo::store(parse_command_line(argc, argv, options), vm);
    if (vm.count("help")) {
      std::cout << options << std::endl;
      return 1;
    }
    bpo::notify(vm);
  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing command line arguments. Available options:\n";

    std::cerr << options << std::endl;
    return 2;
  }

  std::cout << "Generating " << vm["inputdir"].as<std::string>() + vm["name"].as<std::string>() << std::endl;
  alpideResponse(vm["inputdir"].as<std::string>(), vm["outputdir"].as<std::string>(), vm["name"].as<std::string>());

  return 0;
}