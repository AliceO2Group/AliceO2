#include "FairMQLogger.h"
#include "runSimpleMQStateMachine.h"

#include "MFTReconstruction/FindHits.h"
#include "MFTReconstruction/FindTracks.h"
#include "MFTReconstruction/devices/TaskProcessor.h"

using namespace std;
using namespace boost::program_options;
using namespace AliceO2::MFT;

using HitFinder   = TaskProcessor<FindHits>;
using TrackFinder = TaskProcessor<FindTracks>;

int main(int argc, char** argv)
{

  printf("Run MFT reconstruction task processor!\n");

  try {

    std::string taskname;
    std::string keepdata;
    std::string inChannel;
    std::string outChannel;
    
    options_description processor_options("Processor options");
    processor_options.add_options()
      ("task-name",   po::value<std::string>(&taskname)->required()                  ,  "Name of task to run")
      ("keep-data",   po::value<std::string>(&keepdata)                              ,  "Name of data to keep in stream")
      ("in-channel",  po::value<std::string>(&inChannel)->default_value("data-in")   , "input channel name")
      ("out-channel", po::value<std::string>(&outChannel)->default_value("data-out") , "output channel name");
    
    FairMQProgOptions config;
    config.AddToCmdLineOptions(processor_options);
    config.ParseAll(argc, argv);
    
    string control = config.GetValue<std::string>("control");
    LOG(INFO) << "Run::TaskProcessor >>>>> device control is " << control.c_str() << "";
    
    if (strcmp(taskname.c_str(),"FindHits") == 0) {
      HitFinder processor;
      processor.SetDataToKeep(keepdata);
      processor.SetInputChannelName (inChannel);
      processor.SetOutputChannelName(outChannel);
      
      runStateMachine(processor, config);
      /*
      processor.CatchSignals();

      processor.SetConfig(config);

      processor.ChangeState("INIT_DEVICE");
      processor.WaitForEndOfState("INIT_DEVICE");
	
      processor.ChangeState("INIT_TASK");
      processor.WaitForEndOfState("INIT_TASK");
	
      processor.ChangeState("RUN");
      processor.InteractiveStateLoop();
      */
    } else if (strcmp(taskname.c_str(),"FindTracks") == 0) {
      TrackFinder processor;
      processor.SetDataToKeep(keepdata);
      processor.SetInputChannelName (inChannel);
      processor.SetOutputChannelName(outChannel);
      
      runStateMachine(processor, config);
      /*
      processor.CatchSignals();

      processor.SetConfig(config);

      processor.ChangeState("INIT_DEVICE");
      processor.WaitForEndOfState("INIT_DEVICE");
	
      processor.ChangeState("INIT_TASK");
      processor.WaitForEndOfState("INIT_TASK");
	
      processor.ChangeState("RUN");
      processor.InteractiveStateLoop();
      */
    } else {
      LOG(INFO) << "TASK \"" << taskname << "\" UNKNOWN!!!";
    }

  }

  catch (std::exception& e) {

    LOG(ERROR)  << "Unhandled Exception reached the top of main: "
		<< e.what() << ", application will now exit";
    return 1;
    
  }
  
  return 0;
  
}
