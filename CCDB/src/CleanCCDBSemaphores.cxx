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

#include "CCDB/CcdbApi.h"
#include <iostream>
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[], bpo::variables_map& vm)
{
  options.add_options()(
    "cachepath,p", bpo::value<std::string>()->default_value("ccdb"), "path to whole CCDB cache dir as a basis for semaphore search")(
    "sema,s", bpo::value<std::string>()->default_value(""), "Specific named semaphore to be remove")(
    "help,h", "Produce help message.");

  try {
    bpo::store(parse_command_line(argc, argv, options), vm);
    // help
    if (vm.count("help")) {
      std::cout << options << std::endl;
      return false;
    }
    bpo::notify(vm);
  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing command line arguments; Available options:\n";

    std::cerr << options << std::endl;
    return false;
  }
  return true;
}

// A simple tool to clean CCDB related semaphores
int main(int argc, char* argv[])
{
  bpo::options_description options("Tool to find and remove leaking CCDB semaphore from the system");
  bpo::variables_map vm;
  if (!initOptionsAndParse(options, argc, argv, vm)) {
    return 1;
  }

  std::string sema = vm["sema"].as<std::string>();
  if (sema.size() > 0) {
    if (o2::ccdb::CcdbApi::removeSemaphore(sema, true)) {
      std::cout << "Successfully removed " << sema << "\n";
    }
  }

  std::string path = vm["cachepath"].as<std::string>();
  o2::ccdb::CcdbApi::removeLeakingSemaphores(path, true);
  return 0;
}
