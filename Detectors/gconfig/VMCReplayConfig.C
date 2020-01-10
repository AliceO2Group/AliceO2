#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "VMCReplay/VMCReplay.h"
#include "FairRunSim.h"
#include <iostream>
#endif
#include "commonConfig.C"

void Config()
{
  FairRunSim* run = FairRunSim::Instance();
  TString* gModel = run->GetGeoModel();

  // THIS WOULD A GOOD MOMENT TO PASS IN THE CACHED STEP INFORMATION
  auto replayvmc = new VMCReplay("Hello");

  stackSetup(replayvmc, run);

  // ******* replayvmc  specific configuration for simulated Runs  *******
  //
  replayvmc->SetTRIG(1); // Number of events to be processed
  replayvmc->SetSWIT(4, 100);
  replayvmc->SetDEBU(0, 0, 1);

  replayvmc->SetRAYL(1);
  replayvmc->SetSTRA(0);

  // NOTE: Please avoid changing this setting, unless justified as this might lead to very many steps
  // performed by G3; AUTO(1) is the G3 default
  replayvmc->SetAUTO(1); // Select automatic STMIN etc... calc. (AUTO 1) or manual (AUTO 0)

  replayvmc->SetABAN(0); // Restore 3.16 behaviour for abandoned tracks
  replayvmc->SetOPTI(2); // Select optimisation level for GEANT geometry searches (0,1,2)
  replayvmc->SetERAN(5.e-7);
  replayvmc->SetCKOV(1); // cerenkov photons

  // allow many steps per track (per volume)
  // since this is needed in the TPC
  // (this does not seem to be possible per module)
  replayvmc->SetMaxNStep(1E5);
}
