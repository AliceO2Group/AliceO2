/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

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
///    Hadron = FTFP_BERT FTFP_BERT_TRV FTFP_BERT_HP FTFP_INCLXX FTFP_INCLXX_HP FTF_BIC LBE QBBC QGSP_BERT QGSP_BERT_HP QGSP_BIC QGSP_BIC_HP QGSP_FTFP_BERT QGSP_INCLXX QGSP_INCLXX_HP QGS_BIC Shielding ShieldingLEND
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

   // Geant4 VMC 2.x
   // TG4RunConfiguration* runConfiguration
   //         = new TG4RunConfiguration("geomRoot", "QGSP_FTFP_BERT", "stepLimiter+specialCuts+specialControls+stackPopper");

   //Geant4 VMC 3.x
   Bool_t mtMode = false;
   TG4RunConfiguration* runConfiguration
    = new TG4RunConfiguration("geomRoot", "QGSP_FTFP_BERT", "stepLimiter+specialCuts",
                              false, mtMode);

/// Create the G4 VMC
   TGeant4* geant4 = new TGeant4("TGeant4", "The Geant4 Monte Carlo", runConfiguration);
   cout << "Geant4 has been created." << endl;

/// create the Specific stack
   o2::Data::Stack *stack = new AliceO2::Data::Stack(1000);
   stack->StoreSecondaries(kTRUE);
   stack->SetMinPoints(0);
   geant4->SetStack(stack);

   if(FairRunSim::Instance()->IsExtDecayer()){
      TVirtualMCDecayer* decayer = TPythia6Decayer::Instance();
      geant4->SetExternalDecayer(decayer);
   }

/// Customise Geant4 setting
/// (verbose level, global range cut, ..)

   TString configm(gSystem->Getenv("VMCWORKDIR"));
   configm1 = configm + "/Detectors/gconfig/g4config.in";

   //set geant4 specific stuff
  // geant4->SetMaxNStep(10000);  // default is 30000
        // causes failure !! (TO DO: investigate)
  geant4->ProcessGeantMacro(configm1.Data());

  cout << "g4Config.C finished" << endl;
}
