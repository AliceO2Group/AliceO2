///
/// @file    main.cxx
/// @author  Barthelemy von Haller
///

#include "ExampleModule1/Foo.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

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
  o2::Examples::ExampleModule1::Foo hello;
  hello.greet();

  return EXIT_SUCCESS;
}
