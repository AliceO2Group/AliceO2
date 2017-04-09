#include "runFairMQDevice.h"

#include "MFTReconstruction/FindHits.h"
#include "MFTReconstruction/FindTracks.h"
#include "MFTReconstruction/devices/TaskProcessor.h"

using namespace o2::MFT;

using HitFinder   = TaskProcessor<FindHits>;
using TrackFinder = TaskProcessor<FindTracks>;

namespace bpo = boost::program_options;

//_____________________________________________________________________________
void addCustomOptions(bpo::options_description& options)
{

  options.add_options()
    ("task-name",bpo::value<std::string>()->required(),"Name of task to run")
    ("keep-data",bpo::value<std::string>(),"Name of data to keep in stream")
    ("in-channel",bpo::value<std::string>()->default_value("data-in"),"input channel name")
    ("out-channel",bpo::value<std::string>()->default_value("data-out"),"output channel name");

}

//_____________________________________________________________________________
FairMQDevicePtr getDevice(const FairMQProgOptions& config)
{
  std::string taskname = config.GetValue<std::string>("task-name");
  
  LOG(INFO) << "Run::getDevice >>>>> get device with setting!" << "";

  if (strcmp(taskname.c_str(),"FindHits") == 0) {
    LOG(INFO) << "Run::getDevice >>>>> HitFinder" << "";
    return new HitFinder();
  }
  if (strcmp(taskname.c_str(),"FindTracks") == 0) {
    LOG(INFO) << "Run::getDevice >>>>> TrackFinder" << "";
    return new TrackFinder();
  }

  return nullptr;
  
}

