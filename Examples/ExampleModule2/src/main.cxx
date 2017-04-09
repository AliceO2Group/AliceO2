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
