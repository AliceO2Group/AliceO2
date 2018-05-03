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
  using o2::Base::MaterialManager;
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
  // En- or disable setting of special cuts depending on ENV variable \note temorary

  // \note Enable and disable via env variable. Change this or completely get rid of
  //       the possibility to disabled cuts in general?
  if (!getenv("SPECIALPROCSCUTS")) {
    mgr.enableSpecialProcesses(false);
    mgr.enableSpecialCuts(false);
  }

  LOG(INFO) << "Set default settings for processes and cuts.";
  // provide available processes
  const MaterialManager::EProc pair = MaterialManager::kPAIR; /** pair production */
  const MaterialManager::EProc comp = MaterialManager::kCOMP; /** Compton scattering */
  const MaterialManager::EProc phot = MaterialManager::kPHOT; /** photo electric effect */
  const MaterialManager::EProc pfis = MaterialManager::kPFIS; /** photofission */
  const MaterialManager::EProc dray = MaterialManager::kDRAY; /** delta ray */
  const MaterialManager::EProc anni = MaterialManager::kANNI; /** annihilation */
  const MaterialManager::EProc brem = MaterialManager::kBREM; /** bremsstrahlung */
  const MaterialManager::EProc hadr = MaterialManager::kHADR; /** hadronic process */
  const MaterialManager::EProc munu = MaterialManager::kMUNU; /** muon nuclear interaction */
  const MaterialManager::EProc dcay = MaterialManager::kDCAY; /** decay */
  const MaterialManager::EProc loss = MaterialManager::kLOSS; /** energy loss */
  const MaterialManager::EProc muls = MaterialManager::kMULS; /** multiple scattering */
  const MaterialManager::EProc ckov = MaterialManager::kCKOV; /** Cherenkov */
  mgr.DefaultProcesses({ { pair, 1 },
                         { comp, 1 },
                         { phot, 1 },
                         { pfis, 0 },
                         { dray, 0 },
                         { anni, 1 },
                         { brem, 1 },
                         { hadr, 1 },
                         { munu, 1 },
                         { dcay, 1 },
                         { loss, 2 },
                         { muls, 1 },
                         { ckov, 1 } });

  // provide available cuts
  const MaterialManager::ECut cutgam = MaterialManager::kCUTGAM; /** gammas */
  const MaterialManager::ECut cutele = MaterialManager::kCUTELE; /** electrons */
  const MaterialManager::ECut cutneu = MaterialManager::kCUTNEU; /** neutral hadrons */
  const MaterialManager::ECut cuthad = MaterialManager::kCUTHAD; /** charged hadrons */
  const MaterialManager::ECut cutmuo = MaterialManager::kCUTMUO; /** muons */
  const MaterialManager::ECut bcute = MaterialManager::kBCUTE;   /** electron bremsstrahlung */
  const MaterialManager::ECut bcutm = MaterialManager::kBCUTM;   /** muon and hadron bremsstrahlung */
  const MaterialManager::ECut dcute = MaterialManager::kDCUTE;   /** delta-rays by electrons */
  const MaterialManager::ECut dcutm = MaterialManager::kDCUTM;   /** delta-rays by muons */
  const MaterialManager::ECut ppcutm = MaterialManager::kPPCUTM; /** direct pair production by muons */
  const MaterialManager::ECut tofmax = MaterialManager::kTOFMAX; /** time of flight */

  Double_t cut1 = 1.0E-3;         // GeV --> 1 MeV
  Double_t cutb = 1.0E4;          // GeV --> 10 TeV
  Double_t cutTofmax = 1.E10;     // seconds

  mgr.DefaultCuts({ { cutgam, cut1 },
                    { cutele, cut1 },
                    { cutneu, cut1 },
                    { cuthad, cut1 },
                    { cutmuo, cut1 },
                    { bcute, cut1 },
                    { bcutm, cut1 },
                    { dcute, cut1 },
                    { dcutm, cut1 },
                    { ppcutm, cut1 },
                    { tofmax, cutTofmax } });

  const char* settingProc = mgr.specialProcessesEnabled() ? "enabled" : "disabled";
  const char* settingCut = mgr.specialCutsEnabled() ? "enabled" : "disabled";
  LOG(INFO) << "Special process settings are " << settingProc << ".";
  LOG(INFO) << "Special cut settings are " << settingCut << ".";
}
