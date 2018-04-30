/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

/** Configuration macro for setting common cuts and processes for G3, G4 and Fluka (M. Al-Turany 27.03.2008)
    specific cuts and processes to g3 or g4 should be set in the g3Config.C, g4Config.C or flConfig.C

*/

void SetCuts()
{
  cout << "SetCuts Macro: Setting Processes.." <<endl;
   
  // ------>>>> IMPORTANT!!!!
  // For a correct comparison between GEANE and MC (pull distributions) 
  // or for a simulation without the generation of secondary particles:
  // 1. set LOSS = 2, DRAY = 0, BREM = 1
  // 2. set the following cut values: CUTGAM, CUTELE, CUTNEU, CUTHAD, CUTMUO = 1 MeV or less
  //                                  BCUTE, BCUTM, DCUTE, DCUTM, PPCUTM = 10 TeV
  // (For an explanation of the chosen values, please refer to the GEANT User's Guide
  // or to message #5362 in the PandaRoot Forum >> Monte Carlo Engines >> g3Config.C thread)
  // 
  // The default settings refer to a complete simulation which generates and follows also the secondary particles.

  // \note All following settings could also be set in Cave since it is always loaded.
  // Use MaterialManager to set processes and cuts
  auto& mgr = MaterialManager::Instance();

  LOG(INFO) << "Set default settings for processes and cuts.";
  mgr.DefaultProcesses({ { EProc::kPAIR, 1 },    /** pair production */
                         { EProc::kCOMP, 1 },    /** Compton scattering */
                         { EProc::kPHOT, 1 },    /** photo electric effect */
                         { EProc::kPFIS, 0 },    /** photofission */
                         { EProc::kDRAY, 0 },    /** delta ray */
                         { EProc::kANNI, 1 },    /** annihilation */
                         { EProc::kBREM, 1 },    /** bremsstrahlung */
                         { EProc::kHADR, 1 },    /** hadronic process */
                         { EProc::kMUNU, 1 },    /** muon nuclear interaction */
                         { EProc::kDCAY, 1 },    /** decay */
                         { EProc::kLOSS, 2 },    /** energy loss */
                         { EProc::kMULS, 1 },    /** multiple scattering */
                         { EProc::kCKOV, 1 } }); /** Cherenkov */

  const Double_t cut1 = 1.0E-3; // GeV --> 1 MeV
  //Double_t cutb = 1.0E4;          // GeV --> 10 TeV
  const Double_t cutTofmax = 1.E10; // seconds

  mgr.DefaultCuts({ { ECut::kCUTGAM, cut1 },         /** gammas */
                    { ECut::kCUTELE, cut1 },         /** electrons */
                    { ECut::kCUTNEU, cut1 },         /** neutral hadrons */
                    { ECut::kCUTHAD, cut1 },         /** charged hadrons */
                    { ECut::kCUTMUO, cut1 },         /** muons */
                    { ECut::kBCUTE, cut1 },          /** electron bremsstrahlung */
                    { ECut::kBCUTM, cut1 },          /** muon and hadron bremsstrahlung */
                    { ECut::kDCUTE, cut1 },          /** delta-rays by electrons */
                    { ECut::kDCUTM, cut1 },          /** delta-rays by muons */
                    { ECut::kPPCUTM, cut1 },         /** direct pair production by muons */
                    { ECut::kTOFMAX, cutTofmax } }); /** time of flight */

  const char* settingProc = mgr.specialProcessesEnabled() ? "enabled" : "disabled";
  const char* settingCut = mgr.specialCutsEnabled() ? "enabled" : "disabled";
  LOG(INFO) << "Special process settings are " << settingProc << ".";
  LOG(INFO) << "Special cut settings are " << settingCut << ".";
}
