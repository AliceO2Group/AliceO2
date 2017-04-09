///
/// @file   runSim.cxx
/// @author Sebastian Klewin
///

#include <boost/program_options.hpp>
#include <iostream>
#include <sys/wait.h>

#include "../../../../macro/run_sim_tpc.C"
#include "../../../../macro/run_digi_tpc.C"
#include "../../../../macro/run_clus_tpc.C"

namespace bpo = boost::program_options;

int main(int argc, char *argv[])
{
  // Arguments parsing
  bpo::variables_map vm; 
  bpo::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message.")
    ("mode,m",      bpo::value<std::string>()->default_value("sim"),    R"(mode of processing, "sim", "digi", "clus" or "all".)")
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
  std::cout << "#### Starting TPC simulation tool for" << std::endl;
  std::cout << "#### " << events << " events and " << engine << " as MC engine" << std::endl;
  std::cout << "####" << std::endl;
  std::cout << std::endl << std::endl;


  if (mode == "sim") {
    run_sim_tpc(events,engine);
  } else if (mode == "digi") {
    run_digi_tpc(events,engine);
  } else if (mode == "clus") {
    run_clus_tpc(events,engine);
  } else if (mode == "all") {
    int status;
    pid_t PID = fork();
    if (PID == -1) { std::cout << "ERROR" << std::endl; return EXIT_FAILURE;}
    if (PID == 0)  { run_sim_tpc(events,engine); return EXIT_SUCCESS;}
    else waitpid(PID,&status,0);
    
    PID = fork();
    if (PID == -1) { std::cout << "ERROR" << std::endl; return EXIT_FAILURE;}
    if (PID == 0)  { run_digi_tpc(events,engine); return EXIT_SUCCESS;}
    else waitpid(PID,&status,0);

    PID = fork();
    if (PID == -1) { std::cout << "ERROR" << std::endl; return EXIT_FAILURE;}
    if (PID == 0)  { run_clus_tpc(events,engine); return EXIT_SUCCESS;}
    else waitpid(PID,&status,0);
  } else {
      std::cout << "Mode was not recognised" << std::endl;
      return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
