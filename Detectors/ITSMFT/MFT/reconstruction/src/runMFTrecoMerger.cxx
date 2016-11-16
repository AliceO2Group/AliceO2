#include <iostream>

#include "FairMQLogger.h"
#include "FairMQParser.h"
#include "FairMQProgOptions.h"
#include "runSimpleMQStateMachine.h"

#include "MFTReconstruction/devices/Merger.h"

using namespace std;
using namespace boost::program_options;
using namespace AliceO2::MFT;

int main(int argc, char **argv)
{

  printf("Run MFT reconstruction merger!\n");

  try {

    std::string filename;
    std::vector<std::string> classname;
    std::vector<std::string> branchname;
    
    options_description merger_options("Merger options");
    merger_options.add_options()
      ("file-name",   po::value<std::string>             (&filename)  , "Path to the output file")
      ("class-name",  po::value<std::vector<std::string>>(&classname) , "class name")
      ("branch-name", po::value<std::vector<std::string>>(&branchname), "branch name");
    
    
    FairMQProgOptions config;
    config.AddToCmdLineOptions(merger_options);
    
    config.ParseAll(argc, argv);
    
    Merger merger;
    
    runStateMachine(merger, config);
    
  }

  catch (std::exception& e) {

    LOG(ERROR) << "Catch exception! " << e.what() << "";
    
    return 1;

  }
  
  return 0;
  
}
   
