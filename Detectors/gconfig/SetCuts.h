// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
  cout << "SetCuts Macro: Setting Processes.." << endl;

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
  auto& params = GlobalProcessCutSimParam::Instance();

  LOG(INFO) << "Set default settings for processes and cuts.";
  mgr.DefaultProcesses({{EProc::kPAIR, params.PAIR},   /** pair production */
                        {EProc::kCOMP, params.COMP},   /** Compton scattering */
                        {EProc::kPHOT, params.PHOT},   /** photo electric effect */
                        {EProc::kPFIS, params.PFIS},   /** photofission */
                        {EProc::kDRAY, params.DRAY},   /** delta ray */
                        {EProc::kANNI, params.ANNI},   /** annihilation */
                        {EProc::kBREM, params.BREM},   /** bremsstrahlung */
                        {EProc::kHADR, params.HADR},   /** hadronic process */
                        {EProc::kMUNU, params.MUNU},   /** muon nuclear interaction */
                        {EProc::kDCAY, params.DCAY},   /** decay */
                        {EProc::kLOSS, params.LOSS},   /** energy loss */
                        {EProc::kMULS, params.MULS},   /** multiple scattering */
                        {EProc::kCKOV, params.CKOV}}); /** Cherenkov */

  mgr.DefaultCuts({{ECut::kCUTGAM, params.CUTGAM},   /** gammas */
                   {ECut::kCUTELE, params.CUTELE},   /** electrons */
                   {ECut::kCUTNEU, params.CUTNEU},   /** neutral hadrons */
                   {ECut::kCUTHAD, params.CUTHAD},   /** charged hadrons */
                   {ECut::kCUTMUO, params.CUTMUO},   /** muons */
                   {ECut::kBCUTE, params.BCUTE},     /** electron bremsstrahlung */
                   {ECut::kBCUTM, params.BCUTM},     /** muon and hadron bremsstrahlung */
                   {ECut::kDCUTE, params.DCUTE},     /** delta-rays by electrons */
                   {ECut::kDCUTM, params.DCUTM},     /** delta-rays by muons */
                   {ECut::kPPCUTM, params.PPCUTM},   /** direct pair production by muons */
                   {ECut::kTOFMAX, params.TOFMAX}}); /** time of flight */

  const char* settingProc = mgr.specialProcessesEnabled() ? "enabled" : "disabled";
  const char* settingCut = mgr.specialCutsEnabled() ? "enabled" : "disabled";
  LOG(INFO) << "Special process settings are " << settingProc << ".";
  LOG(INFO) << "Special cut settings are " << settingCut << ".";
  mgr.printProcesses();
  mgr.printCuts();
}
