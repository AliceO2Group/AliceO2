// Configuration macro for MCReplay VirtualMC

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "MCReplay/MCReplayEngine.h"
#include "SimSetup/MCReplayParam.h"
#include "FairRunSim.h"
#endif
#include "commonConfig.C"

void Config()
{
  // TString* gModel = run->GetGeoModel();
  FairRunSim* run = FairRunSim::Instance();
  auto* replay = new mcreplay::MCReplayEngine();
  stackSetup(replay, run);
  auto& params = o2::MCReplayParam::Instance();
  replay->setStepFilename(params.stepFilename);
  replay->setStepTreename(params.stepTreename);
  replay->SetCut("CUTALLE", params.energyCut);
}
