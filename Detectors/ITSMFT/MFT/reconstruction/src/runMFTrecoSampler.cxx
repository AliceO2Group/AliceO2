#include <iostream>

#include "TApplication.h"

#include "FairMQLogger.h"
#include "FairMQParser.h"
#include "FairMQProgOptions.h"
#include "runSimpleMQStateMachine.h"

#include "MFTBase/EventHeader.h"
#include "MFTReconstruction/FindHits.h"
#include "MFTReconstruction/devices/Sampler.h"

using namespace std;
using namespace boost::program_options;
using namespace AliceO2::MFT;

int main(int argc, char **argv)
{

  printf("Run MFT reconstruction sampler!\n");

  try {

    std::vector<std::string> filename;
    std::vector<std::string> branchname;
    
    options_description sampler_options("MFT sampler options");
    sampler_options.add_options()
      ("file-name", value<std::vector<std::string>>(&filename),
       "Path to the input file")
      ("branch-name", value<std::vector<std::string>>(&branchname)->required(), "branch name");
    
    FairMQProgOptions config;
    config.AddToCmdLineOptions(sampler_options);
    config.ParseAll(argc, argv);
    
    string control = config.GetValue<std::string>("control");
    LOG(INFO) << "Run::Sampler >>>>> device control is " << control.c_str() << "";
    
    LOG(INFO) << "Run::Sampler >>>>> Using file " << filename.at(0).c_str() << " and branch " << branchname.at(0).c_str() << "";
    
    Sampler sampler;
    
    for (UInt_t ielem = 0; ielem < filename.size(); ielem++) {
      sampler.AddInputFileName(filename.at(ielem));
    }
      
    for (UInt_t ielem = 0; ielem < branchname.size(); ielem++) {
      sampler.AddInputBranchName(branchname.at(ielem));
      LOG(INFO) << "Run::Sampler >>>>> add input branch " << branchname.at(ielem).c_str() << "";
   }
      
    if (strcmp(branchname.at(0).c_str(),"MFTPoints") == 0) {
      LOG(INFO) << "Run::Sampler >>>>> add input branch MCEventHeader." << "";
      sampler.AddInputBranchName("MCEventHeader.");
    }

    if (strcmp(branchname.at(0).c_str(),"MFTHits") == 0) {
      LOG(INFO) << "Run::Sampler >>>>> add input branch EventHeader." << "";
      sampler.AddInputBranchName("EventHeader.");
    }

    TApplication app("Sampler", 0, 0);
      
    runStateMachine(sampler, config);
    /*
    sampler.CatchSignals();
	
    sampler.SetConfig(config);
	
    sampler.ChangeState("INIT_DEVICE");
    sampler.WaitForEndOfState("INIT_DEVICE");
    
    sampler.ChangeState("INIT_TASK");
    sampler.WaitForEndOfState("INIT_TASK");
    
    sampler.ChangeState("RUN");
    sampler.InteractiveStateLoop();
    */
    gApplication->Terminate();

  } 

  catch (std::exception& e) {
    
    LOG(ERROR)  << "Unhandled Exception reached the top of main: " 
		<< e.what() << ", application will now exit";
    
    return 1;

  }
  
  return 0;
  
}

