/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include "TGeant4.h"
#include "TString.h"
#include "TPythia6Decayer.h"
#include "FairRunSim.h"
#include "TSystem.h"
#include "TG4RunConfiguration.h"
#endif
#include "commonConfig.C"

// Configuration macro for Geant4 VirtualMC
void Config()
{
  ///    Create the run configuration
  /// In constructor user has to specify the geometry input
  /// and select geometry navigation via the following options:
  /// - geomVMCtoGeant4   - geometry defined via VMC, G4 native navigation
  /// - geomVMCtoRoot     - geometry defined via VMC, Root navigation
  /// - geomRoot          - geometry defined via Root, Root navigation
  /// - geomRootToGeant4  - geometry defined via Root, G4 native navigation
  /// - geomGeant4        - geometry defined via Geant4, G4 native navigation
  ///
  /// The second argument in the constructor selects physics list:
  ///    Available options:
  ///    EMonly, EMonly+Extra, Hadron_EM, Hadron_EM+Extra
  ///    where EMonly = emStandard
  ///    Hadron = FTFP_BERT FTFP_BERT_TRV FTFP_BERT_HP FTFP_INCLXX FTFP_INCLXX_HP FTF_BIC LBE QBBC QGSP_BERT
  ///             QGSP_BERT_HP QGSP_BIC QGSP_BIC_HP QGSP_FTFP_BERT QGSP_INCLXX QGSP_INCLXX_HP QGS_BIC
  ///	          Shielding ShieldingLEND
  ///    EM =  _EMV _EMX _EMY _EMZ _LIV _PEN
  ///    Extra = extra optical radDecay
  ///    The Extra selections are cumulative, while Hadron selections are exlusive.

  /// The third argument activates the special processes in the TG4SpecialPhysicsList,
  /// which implement VMC features:
  /// - stepLimiter       - step limiter (default)
  /// - specialCuts       - VMC cuts
  /// - specialControls   - VMC controls for activation/inactivation selected processes
  /// - stackPopper       - stackPopper process
  /// When more than one options are selected, they should be separated with '+'
  /// character: eg. stepLimit+specialCuts.

  //Geant4 VMC 3.x
  Bool_t mtMode = FairRunSim::Instance()->IsMT();
  Bool_t specialStacking = true; // leads to default stack behaviour in which new primaries are only started if the previous
                                 // one and all of its secondaries have been transported
                                 // any other choice is dangerously inconsistent with the FinishPrimary() interface of VMCApp

  auto runConfiguration = new TG4RunConfiguration("geomRoot", "QGSP_FTFP_BERT+optical+biasing", "stepLimiter+specialCuts",
                                                  specialStacking, mtMode);

  /// Create the G4 VMC
  TGeant4* geant4 = new TGeant4("TGeant4", "The Geant4 Monte Carlo", runConfiguration);
  cout << "Geant4 has been created." << endl;

  // setup the stack
  stackSetup(geant4, FairRunSim::Instance());

  // setup decayer
  if (FairRunSim::Instance()->IsExtDecayer()) {
    TVirtualMCDecayer* decayer = TPythia6Decayer::Instance();
    geant4->SetExternalDecayer(decayer);
  }

  TString configm(gSystem->Getenv("VMCWORKDIR"));
  auto configm1 = configm + "/Detectors/gconfig/g4config.in";

  /// Customise Geant4 setting
  /// (verbose level, global range cut, ..)
  geant4->ProcessGeantMacro(configm1.Data());

  // Enter in Geant4 Interactive mode
  // geant4->StartGeantUI();

  cout << "g4Config.C finished" << endl;
}
