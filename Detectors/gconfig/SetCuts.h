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
  

  TVirtualMC::GetMC()->SetProcess("PAIR",1); /** pair production*/
  TVirtualMC::GetMC()->SetProcess("COMP",1); /**Compton scattering*/
  TVirtualMC::GetMC()->SetProcess("PHOT",1); /** photo electric effect */
  TVirtualMC::GetMC()->SetProcess("PFIS",0); /**photofission*/
  TVirtualMC::GetMC()->SetProcess("DRAY",0); /**delta-ray*/
  TVirtualMC::GetMC()->SetProcess("ANNI",1); /**annihilation*/
  TVirtualMC::GetMC()->SetProcess("BREM",1); /**bremsstrahlung*/
  TVirtualMC::GetMC()->SetProcess("HADR",1); /**hadronic process*/
  TVirtualMC::GetMC()->SetProcess("MUNU",1); /**muon nuclear interaction*/
  TVirtualMC::GetMC()->SetProcess("DCAY",1); /**decay*/
  TVirtualMC::GetMC()->SetProcess("LOSS",2); /**energy loss*/
  TVirtualMC::GetMC()->SetProcess("MULS",1); /**multiple scattering*/
  TVirtualMC::GetMC()->SetProcess("CKOV",1); /**cherenkov */

  
    
  
  Double_t cut1 = 1.0E-3;         // GeV --> 1 MeV
  Double_t cutb = 1.0E4;          // GeV --> 10 TeV
  Double_t tofmax = 1.E10;        // seconds
  cout << "SetCuts Macro: Setting cuts.." <<endl;
  
  TVirtualMC::GetMC()->SetCut("CUTGAM",cut1);   /** gammas (GeV)*/
  TVirtualMC::GetMC()->SetCut("CUTELE",cut1);   /** electrons (GeV)*/
  TVirtualMC::GetMC()->SetCut("CUTNEU",cut1);   /** neutral hadrons (GeV)*/
  TVirtualMC::GetMC()->SetCut("CUTHAD",cut1);   /** charged hadrons (GeV)*/
  TVirtualMC::GetMC()->SetCut("CUTMUO",cut1);   /** muons (GeV)*/
  TVirtualMC::GetMC()->SetCut("BCUTE",cut1);    /** electron bremsstrahlung (GeV)*/
  TVirtualMC::GetMC()->SetCut("BCUTM",cut1);    /** muon and hadron bremsstrahlung(GeV)*/ 
  TVirtualMC::GetMC()->SetCut("DCUTE",cut1);    /** delta-rays by electrons (GeV)*/
  TVirtualMC::GetMC()->SetCut("DCUTM",cut1);    /** delta-rays by muons (GeV)*/
  TVirtualMC::GetMC()->SetCut("PPCUTM",cut1);   /** direct pair production by muons (GeV)*/
  TVirtualMC::GetMC()->SetCut("TOFMAX",tofmax); /**time of flight cut in seconds*/
  
   
}
