// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file    main.cxx
/// @author  Barthelemy von Haller
///

#include "ExampleModule2/Foo.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <TClass.h>

namespace po = boost::program_options;
using namespace std;

int main(int argc, char *argv[])
{
  // Arguments parsing
  po::variables_map vm;
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message.");
  po::store(parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  // help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }

  // Actual "work"
  o2::Examples::ExampleModule2::Foo hello;
  hello.greet();
  std::cout << "Class is " << hello.Class()->GetName() << std::endl;

  return EXIT_SUCCESS;
}
