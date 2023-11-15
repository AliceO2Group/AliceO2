/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
R__LOAD_LIBRARY(libG4ptl)
R__LOAD_LIBRARY(libG4zlib)
R__LOAD_LIBRARY(libG4expat)
R__LOAD_LIBRARY(libG4clhep)
R__LOAD_LIBRARY(libG4tools)
R__LOAD_LIBRARY(libG4global)
R__LOAD_LIBRARY(libG4intercoms)
R__LOAD_LIBRARY(libG4graphics_reps)
R__LOAD_LIBRARY(libG4materials)
R__LOAD_LIBRARY(libG4geometry)
R__LOAD_LIBRARY(libG4particles)
R__LOAD_LIBRARY(libG4track)
R__LOAD_LIBRARY(libG4digits_hits)
R__LOAD_LIBRARY(libG4analysis)
R__LOAD_LIBRARY(libG4processes)
R__LOAD_LIBRARY(libG4parmodels)
R__LOAD_LIBRARY(libG4tracking)
R__LOAD_LIBRARY(libG4event)
R__LOAD_LIBRARY(libG4run)
R__LOAD_LIBRARY(libG4physicslists)
R__LOAD_LIBRARY(libG4readout)
R__LOAD_LIBRARY(libG4error_propagation)
R__LOAD_LIBRARY(libG4persistency)
R__LOAD_LIBRARY(libG4interfaces)
R__LOAD_LIBRARY(libG3toG4)
R__LOAD_LIBRARY(libG4tasking)
R__LOAD_LIBRARY(libG4modeling)
R__LOAD_LIBRARY(libG4vis_management)
R__LOAD_LIBRARY(libG4VRML)
R__LOAD_LIBRARY(libG4RayTracer)
R__LOAD_LIBRARY(libG4visHepRep)
R__LOAD_LIBRARY(libG4GMocren)
R__LOAD_LIBRARY(libG4FR)
R__LOAD_LIBRARY(libG4Tree)
R__LOAD_LIBRARY(libClhepVGM)
R__LOAD_LIBRARY(libBaseVGM)
R__LOAD_LIBRARY(libGeant4GM)
R__LOAD_LIBRARY(libRootGM)
R__LOAD_LIBRARY(libXmlVGM)
R__LOAD_LIBRARY(libVMCLibrary)
R__LOAD_LIBRARY(libg4root)
R__LOAD_LIBRARY(libgeant4vmc)

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include "TGeant4.h"
#include "TString.h"
#include "FairRunSim.h"
#include "TSystem.h"
#include "TG4RunConfiguration.h"
#include "SimConfig/G4Params.h"
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
  ///                  Shielding ShieldingLEND
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

  auto& g4Params = ::o2::conf::G4Params::Instance();
  auto& physicsSetup = g4Params.getPhysicsConfigString();
  std::cout << "PhysicsSetup wanted " << physicsSetup << "\n";
  auto runConfiguration = new TG4RunConfiguration("geomRoot", physicsSetup, "stepLimiter+specialCuts",
                                                  specialStacking, mtMode);
  /// avoid the use of G4BACKTRACE (it seems to inferfere with process logic in o2-sim)
  setenv("G4BACKTRACE", "none", 1);

  /// Create the G4 VMC
  TGeant4* geant4 = new TGeant4("TGeant4", "The Geant4 Monte Carlo", runConfiguration);
  std::cout << "Geant4 has been created." << std::endl;

  // setup the stack
  stackSetup(geant4, FairRunSim::Instance());

  // setup decayer
  decayerSetup(geant4);

  // eventually apply custom g4config.in
  if (g4Params.configMacroFile.size() > 0) {
    std::cout << "Applying custom config file " << g4Params.configMacroFile << "\n";
    geant4->ProcessGeantMacro(g4Params.configMacroFile.c_str());
  } else {
    TString configm(gSystem->Getenv("VMCWORKDIR"));
    auto configm1 = configm + "/Detectors/gconfig/g4config.in";

    /// Customise Geant4 setting
    /// (verbose level, global range cut, ..)
    geant4->ProcessGeantMacro(configm1.Data());
  }

  // Enter in Geant4 Interactive mode
  // geant4->StartGeantUI();

  std::cout << "g4Config.C finished" << std::endl;
}

void Terminate()
{
  static bool terminated = false;
  if (!terminated) {
    std::cout << "Executing G4 terminate\n";
    TGeant4* geant4 = dynamic_cast<TGeant4*>(TVirtualMC::GetMC());
    if (geant4) {
      // we need to call finish run for Geant4 ... Since we use ProcessEvent() interface;
      geant4->FinishRun();
    }
    terminated = true;
  }
}
