// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   runSim.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

#include <boost/program_options.hpp>
#include <iostream>

namespace bpo = boost::program_options;

int main(int argc, char* argv[])
{
  // Arguments parsing
  bpo::variables_map vm;
  bpo::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message.")("file,f", "input file(s)");
  bpo::store(parse_command_line(argc, argv, desc), vm);
  bpo::notify(vm);

  // help
  if (vm.count("help")) {
    std::cout << desc << '\n';
    return EXIT_SUCCESS;
  }

  // Actual "work"
  const std::string file = vm["file"].as<std::string>();

  std::cout << "####" << '\n';
  std::cout << "#### Starting TPC simple online monitor" << '\n';
  std::cout << "#### filename: " << file << '\n';
  std::cout << "####" << '\n';
  std::cout << '\n'
            << '\n';

  return EXIT_SUCCESS;
}
