// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SetCuts.h"
#include "SimConfig/GlobalProcessCutSimParam.h"
#include "DetectorsBase/MaterialManager.h"
#include <fairlogger/Logger.h>

using namespace o2::base;

namespace o2
{

void SetCuts()
{
  LOG(info) << "Setup global cuts and processes";

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
  auto& mgr = o2::base::MaterialManager::Instance();
  // This loads default cuts and processes if they are defined in the MaterialManagerParam.inputFile
  // The cuts and processes below will only be set if they were not defined in the JSON
  mgr.loadCutsAndProcessesFromJSON();
  auto& params = o2::GlobalProcessCutSimParam::Instance();

  LOG(info) << "Set default settings for processes and cuts.";
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
  LOG(info) << "Special process settings are " << settingProc << ".";
  LOG(info) << "Special cut settings are " << settingCut << ".";
}

} // namespace o2
