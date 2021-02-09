// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// executable to retrieve objects from CCDB emulating the rates that we expect for
// Run 3, as read (in terms of size and rate) from an external file

#include "retrieveFromCCDB.C"
#include <TRandom.h>
#include <boost/program_options.hpp>
#include <iostream>

namespace bpo = boost::program_options;

bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[], bpo::variables_map& vm)
{
  options.add_options()(
    "TFs-in-parallel,m", bpo::value<int>()->default_value(8), "Number of TFs to simulate that access the CCDB in parallel")(
    "TF-processing-time,t", bpo::value<float>()->default_value(10.), "Seconds supposed to be needed to process a TF")(
    "ccdb-sercer,s", bpo::value<std::string>()->default_value("ccdb-test.cern.ch:8080"), "CCDB server")(
    "in-file-name,n", bpo::value<std::string>()->default_value("cdbSizeV0.txt"), "File name with list of CCDB entries to upload")(
    "disable-caching,d", bpo::value<bool>()->default_value(false)->implicit_value(true), "Disable CCDB caching")(
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
  auto nTFs = vm["TFs-in-parallel"].as<int>();
  auto tTF = vm["TF-processing-time"].as<float>();
  auto& inputFile = vm["in-file-name"].as<std::string>();
  auto& ccdbHost = vm["ccdb-sercer"].as<std::string>();
  auto disableCaching = vm["disable-caching"].as<bool>();
  retrieveFromCCDB(nTFs, tTF, inputFile, ccdbHost, disableCaching);

  return (0);
}
