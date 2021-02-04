// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// executable to populate the CCDB emulating the rates that we expect for
// Run 3, as read (in terms of size and rate) from an external file

#include "populateCCDB.C"
#include <TRandom.h>
#include <boost/program_options.hpp>
#include <iostream>

namespace bpo = boost::program_options;

bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[], bpo::variables_map& vm)
{
  options.add_options()(
    "host", bpo::value<std::string>()->default_value("ccdb-test.cern.ch:8080"), "CCDB server")(
    "in-file-name,n", bpo::value<std::string>()->default_value("cdbSizeV0.txt"), "File name with list of CCDB entries to upload")(
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

int main(int argc, char* argv[])
{
  bpo::options_description options("Allowed options");
  bpo::variables_map vm;
  if (!initOptionsAndParse(options, argc, argv, vm)) {
    return -1;
  }

  // call populate "macro"
  auto& inputFile = vm["in-file-name"].as<std::string>();
  auto& ccdbHost = vm["host"].as<std::string>();
  populateCCDB(inputFile, ccdbHost);

  return (0);
}
