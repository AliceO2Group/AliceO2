#include "FairMQLogger.h"
#include "runSimpleMQStateMachine.h"

#include "MFTReconstruction/devices/FileSink.h"

using namespace std;
using namespace boost::program_options;
using namespace AliceO2::MFT;

int main(int argc, char** argv)
{

  FairMQProgOptions config;
  FileSink fileSink;

  try 
    {

      std::string filename;
      std::vector<std::string> classname;
      std::vector<std::string> branchname;
      std::string inChannel;
      
      options_description fileSink_options("FileSink options");
      fileSink_options.add_options()
	("file-name", value<std::string>(&filename), "Path to the output file")
	("class-name", value<std::vector<std::string>>(&classname), "class name")
	("branch-name", value<std::vector<std::string>>(&branchname), "branch name")
	("in-channel", value<std::string>(&inChannel)->default_value("data-in") , "input channel name");

      config.AddToCmdLineOptions(fileSink_options);
      
      config.ParseAll(argc, argv);

      fileSink.SetProperty(FileSink::OutputFileName,filename);

      if ( classname.size() != branchname.size() ) {
	LOG(ERROR) << "Run::FileSink >>>>> The classname size (" << classname.size() << ") and branchname size (" << branchname.size() << ") MISMATCH!!!";
      }
      
      //fileSink.AddOutputBranch("FairEventHeader","EventHeader.");
      for ( unsigned int ielem = 0 ; ielem < classname.size() ; ielem++ ) {
	fileSink.AddOutputBranch(classname.at(ielem),branchname.at(ielem));
      }
      
      fileSink.SetInputChannelName(inChannel);
      
      string control = config.GetValue<std::string>("control");
      LOG(INFO) << "Run::FileSink >>>>> device control is " << control.c_str() << "";

      runStateMachine(fileSink, config);     
      
      // equivalent to this (interactive mode)
      /*
      fileSink.CatchSignals();

      fileSink.SetConfig(config);

      fileSink.ChangeState("INIT_DEVICE");
      fileSink.WaitForEndOfState("INIT_DEVICE");
      
      fileSink.ChangeState("INIT_TASK");
      fileSink.WaitForEndOfState("INIT_TASK");
      
      fileSink.ChangeState("RUN");
      fileSink.InteractiveStateLoop();
      */
    }
  catch (std::exception& e)
    {
      LOG(ERROR)  << "Unhandled Exception reached the top of main: " 
		  << e.what() << ", application will now exit";
      return 1;
    }
  
  return 0;
  
}
