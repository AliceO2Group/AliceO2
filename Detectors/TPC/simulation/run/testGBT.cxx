///
/// @file   runSim.cxx
/// @author Sebastian Klewin
///

#include <boost/program_options.hpp>
#include <iostream>

#include <mutex>
#include <thread>
#include <fstream>
#include "TPCSimulation/GBTFrameContainer.h"
#include "../../../../macro/test_GBTFrame.C"

namespace bpo = boost::program_options;

int main(int argc, char *argv[])
{
  // Arguments parsing
  bpo::variables_map vm; 
  bpo::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message.")
    ("infile,i",    bpo::value<std::string>()->default_value("/misc/alidata120/alice_u/sklewin/alice/fifo_data_0"), "Input data file");
  bpo::store(parse_command_line(argc, argv, desc), vm);
  bpo::notify(vm);

  // help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }

  // Actual "work"
  std::string infile = vm["infile"].as<std::string>();

  test_GBTFrame(infile);

  return EXIT_SUCCESS;
}
