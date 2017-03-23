///
/// @file   runSim.cxx
/// @author Sebastian Klewin
///

#include <boost/program_options.hpp>
#include <iostream>

#include "../../../../macro/run_sim.C"
#include "../../../../macro/run_digi.C"
#include "../../../../macro/run_clusterer.C"
#include "../../../../macro/compare_cluster.C"

namespace bpo = boost::program_options;

int main(int argc, char *argv[])
{
  // Arguments parsing
  bpo::variables_map vm; 
  bpo::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message.")
    ("mode,m",      bpo::value<std::string>()->default_value("sim"),    "mode of processing, \"sim\", \"digi\", \"clusterer\" or \"check\".")
    ("nEvents,n",   bpo::value<int>()->default_value(2),                "number of events to simulate.")
    ("mcEngine,e",  bpo::value<std::string>()->default_value("TGeant3"), "MC generator to be used.");
  bpo::store(parse_command_line(argc, argv, desc), vm);
  bpo::notify(vm);

  // help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }

  // Actual "work"
  const int events = vm["nEvents"].as<int>();
  const std::string engine = vm["mcEngine"].as<std::string>();
  const std::string mode = vm["mode"].as<std::string>();

  std::cout << "####" << std::endl;
  std::cout << "#### Staring TPC simulation tool for" << std::endl;
  std::cout << "#### " << events << " events and " << engine << " as MC engine" << std::endl;
  std::cout << "####" << std::endl;
  std::cout << std::endl << std::endl;


  if (mode == "sim") {
    run_sim(events,engine);
  } else if (mode == "digi") {
    run_digi(events,engine);
  } else if (mode == "clusterer") {
    run_clusterer(events,engine);
  } else if (mode == "check") {
    compare_cluster(events,engine);
  } else {
      std::cout << "Mode was not recognised" << std::endl;
      return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
