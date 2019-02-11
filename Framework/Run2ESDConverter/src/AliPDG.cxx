/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/* $Id$ */

// Class to encapsulate the ALICE updates to TDatabasePDG.h
// Can be used by TGeant3 and TGeant4
// It contains also the constants for the PDG particle IDs.
// Should evolve towards dynamical loading from external data base.
// Comments to: andreas.morsch@cern.ch

#include "AliPDG.h"
#include "TDatabasePDG.h"

ClassImp(AliPDG)



void AliPDG::AddParticlesToPdgDataBase()
{

//
// Add particles to the PDG data base
//

  static Bool_t bAdded = kFALSE;
  // Check if already called
  if(bAdded)return;
  bAdded = true;

  TDatabasePDG *pdgDB = TDatabasePDG::Instance();
  const Int_t kspe=50000000;

  // PDG nuclear states are 10-digit numbers
  // 10LZZZAAAI e.g. deuteron is
  // 1000010020
  const Int_t kion=1000000000;

/*
  const Double_t kAu2Gev=0.9314943228;
*/

  const Double_t khSlash = 1.0545726663e-27;
  const Double_t kErg2Gev = 1/1.6021773349e-3;
  const Double_t khShGev = khSlash*kErg2Gev;
  const Double_t kYear2Sec = 3600*24*365.25;

//
// Bottom mesons
// mass and life-time from PDG
//
  pdgDB->AddParticle("Upsilon(3S)","Upsilon(3S)",10.3552,kTRUE,
                     0,1,"Bottonium",200553);

// QCD diffractive states
  pdgDB->AddParticle("rho_diff0","rho_diff0",0,kTRUE,
		     0,0,"QCD diffr. state",9900110);
  pdgDB->AddParticle("pi_diffr+","pi_diffr+",0,kTRUE,
		     0,1,"QCD diffr. state",9900210);
  pdgDB->AddParticle("omega_di","omega_di",0,kTRUE,
		     0,0,"QCD diffr. state",9900220);
  pdgDB->AddParticle("phi_diff","phi_diff",0,kTRUE,
		     0,0,"QCD diffr. state",9900330);
  pdgDB->AddParticle("J/psi_di","J/psi_di",0,kTRUE,
		     0,0,"QCD diffr. state",9900440);
  pdgDB->AddParticle("n_diffr0","n_diffr0",0,kTRUE,
		     0,0,"QCD diffr. state",9902110);
  pdgDB->AddParticle("p_diffr+","p_diffr+",0,kTRUE,
		     0,1,"QCD diffr. state",9902210);

// From Herwig
  pdgDB->AddParticle("PSID    ", " ", 3.7699, kFALSE, 0.0, 0, "meson",   30443);

  pdgDB->AddParticle("A_00    ", " ", 0.9960, kFALSE, 0.0, 0, "meson",  9000111);
  pdgDB->AddParticle("A_0+    ", " ", 0.9960, kFALSE, 0.0,+3, "meson",  9000211);
  pdgDB->AddParticle("A_0-    ", " ", 0.9960, kFALSE, 0.0,-3, "meson", -9000211);

  //pdgDB->AddParticle("F0P0    ", " ", 0.9960, kFALSE, 0.0, 0, "meson",  9010221);

  pdgDB->AddParticle("KDL_2+  ", " ", 1.773,  kFALSE, 0.0,+3, "meson",   10325);
  pdgDB->AddParticle("KDL_2-  ", " ", 1.773,  kFALSE, 0.0,-3, "meson",  -10325);

  pdgDB->AddParticle("KDL_20  ", " ", 1.773,  kFALSE, 0.0, 0, "meson",   10315);
  pdgDB->AddParticle("KDL_2BR0", " ", 1.773,  kFALSE, 0.0, 0, "meson",  -10315);

  pdgDB->AddParticle("PI_2+   ", " ", 1.670,  kFALSE, 0.0,+3, "meson",   10215);
  pdgDB->AddParticle("PI_2-   ", " ", 1.670,  kFALSE, 0.0,-3, "meson",  -10215);
  pdgDB->AddParticle("PI_20   ", " ", 1.670,  kFALSE, 0.0, 0, "meson",   10115);


  pdgDB->AddParticle("KD*+    ", " ", 1.717,  kFALSE, 0.0,+3, "meson",   30323);
  pdgDB->AddParticle("KD*-    ", " ", 1.717,  kFALSE, 0.0,-3, "meson",  -30323);

  pdgDB->AddParticle("KD*0    ", " ", 1.717,  kFALSE, 0.0, 0, "meson",   30313);
  pdgDB->AddParticle("KDBR*0  ", " ", 1.717,  kFALSE, 0.0, 0, "meson",  -30313);

  pdgDB->AddParticle("RHOD+   ", " ", 1.700,  kFALSE, 0.0,+3, "meson",   30213);
  pdgDB->AddParticle("RHOD-   ", " ", 1.700,  kFALSE, 0.0,-3, "meson",  -30213);
  pdgDB->AddParticle("RHOD0   ", " ", 1.700,  kFALSE, 0.0, 0, "meson",   30113);

  pdgDB->AddParticle("ETA_2(L)", " ", 1.632,  kFALSE, 0.0, 0, "meson",   10225);
  pdgDB->AddParticle("ETA_2(H)", " ", 1.854,  kFALSE, 0.0, 0, "meson",   10335);
  pdgDB->AddParticle("OMEGA(H)", " ", 1.649,  kFALSE, 0.0, 0, "meson",   30223);


  pdgDB->AddParticle("KDH_2+  ", " ", 1.816,  kFALSE, 0.0,+3, "meson",   20325);
  pdgDB->AddParticle("KDH_2-  ", " ", 1.816,  kFALSE, 0.0,-3, "meson",  -20325);

  pdgDB->AddParticle("KDH_20  ", " ", 1.816,  kFALSE, 0.0, 0, "meson",   20315);
  pdgDB->AddParticle("KDH_2BR0", " ", 1.816,  kFALSE, 0.0, 0, "meson",  -20315);


  pdgDB->AddParticle("KD_3+   ", " ", 1.773,  kFALSE, 0.0,+3, "meson",     327);
  pdgDB->AddParticle("KD_3-   ", " ", 1.773,  kFALSE, 0.0,-3, "meson",    -327);

  pdgDB->AddParticle("KD_30   ", " ", 1.773,  kFALSE, 0.0, 0, "meson",     317);
  pdgDB->AddParticle("KD_3BR0 ", " ", 1.773,  kFALSE, 0.0, 0, "meson",    -317);

  pdgDB->AddParticle("RHO_3+  ", " ", 1.691,  kFALSE, 0.0,+3, "meson",     217);
  pdgDB->AddParticle("RHO_3-  ", " ", 1.691,  kFALSE, 0.0,-3, "meson",    -217);
  pdgDB->AddParticle("RHO_30  ", " ", 1.691,  kFALSE, 0.0, 0, "meson",     117);
  pdgDB->AddParticle("OMEGA_3 ", " ", 1.667,  kFALSE, 0.0, 0, "meson",     227);
  pdgDB->AddParticle("PHI_3   ", " ", 1.854,  kFALSE, 0.0, 0, "meson",     337);

  pdgDB->AddParticle("CHI2P_B0", " ", 10.232, kFALSE, 0.0, 0, "meson", 110551);
  pdgDB->AddParticle("CHI2P_B1", " ", 10.255, kFALSE, 0.0, 0, "meson", 120553);
  pdgDB->AddParticle("CHI2P_B2", " ", 10.269, kFALSE, 0.0, 0, "meson", 100555);
  pdgDB->AddParticle("UPSLON4S", " ", 10.580, kFALSE, 0.0, 0, "meson", 300553);


  // IONS
  //
  // Done by default now from Pythia6 table
  // Needed for other generators
  // So check if already defined


  Int_t ionCode = kion+10020;
  if(!pdgDB->GetParticle(ionCode)){
      pdgDB->AddParticle("Deuteron","Deuteron", 1.875613, kTRUE,
			 0,3,"Ion",ionCode);
  }
  pdgDB->AddAntiParticle("AntiDeuteron", - ionCode);

  ionCode = kion+10030;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("Triton","Triton", 2.80925, kFALSE,
                     khShGev/(12.33*kYear2Sec),3,"Ion",ionCode);
  }
  pdgDB->AddAntiParticle("AntiTriton", - ionCode);

  ionCode = kion+20030;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("HE3","HE3", 2.80923,kFALSE,
                     0,6,"Ion",ionCode);
  }
  pdgDB->AddAntiParticle("AntiHE3", - ionCode);

  ionCode = kion+20040;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("Alpha","Alpha", 3.727379, kTRUE,
		       khShGev/(12.33*kYear2Sec), 6, "Ion", ionCode);
  }
  pdgDB->AddAntiParticle("AntiAlpha", - ionCode);

// Special particles
//
  pdgDB->AddParticle("Cherenkov","Cherenkov",0,kFALSE,
		     0,0,"Special",kspe+50);
  pdgDB->AddParticle("FeedbackPhoton","FeedbackPhoton",0,kFALSE,
		     0,0,"Special",kspe+51);
  pdgDB->AddParticle("Lambda1520","Lambda1520",1.5195,kFALSE,
		     0.0156,0,"Resonance",3124);
  pdgDB->AddAntiParticle("Lambda1520bar",-3124);

  //Hyper nuclei and exotica
  ionCode = 1010010030;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("HyperTriton","HyperTriton", 2.99131, kFALSE,
		       2.5e-15, 3, "Ion", ionCode);
  }

  ionCode = -1010010030;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiHyperTriton","AntiHyperTriton", 2.99131, kFALSE,
		       2.5e-15, 3, "Ion", ionCode);
  }

  ionCode = 1010010040;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("Hyperhydrog4","Hyperhydrog4", 3.931, kFALSE,
		       2.5e-15, 3, "Ion", ionCode);
  }

  ionCode = -1010010040;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiHyperhydrog4","AntiHyperhydrog4", 3.931, kFALSE,
		       2.5e-15, 3, "Ion", ionCode);
  }

  ionCode = 1010020040;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("Hyperhelium4","Hyperhelium4", 3.929, kFALSE,
		       2.5e-15, 6, "Ion", ionCode);
  }

  ionCode = -1010020040;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiHyperhelium4","AntiHyperhelium4", 3.929, kFALSE,
		       2.5e-15, 6, "Ion", ionCode);
  }

  ionCode = 1010020050;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("Hyperhelium5","Hyperhelium5", 4.841, kFALSE,
               2.5e-15, 6, "Ion", ionCode);
  }
    
  ionCode = -1010020050;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiHyperhelium5","AntiHyperhelium5", 4.841, kFALSE,
               2.5e-15, 6, "Ion", ionCode);
  }
    
    ionCode = 1020010040;
    if(!pdgDB->GetParticle(ionCode)){
        pdgDB->AddParticle("DoubleHyperhydrogen4","DoubleHyperhydrogen4", 4.106, kFALSE,
                           2.5e-15, 6, "Ion", ionCode);
    }
    
    ionCode = -1020010040;
    if(!pdgDB->GetParticle(ionCode)){
        pdgDB->AddParticle("DoubleAntiHyperhydrogen4","DoubleAntiHyperhydrogen4", 4.106, kFALSE,
                           2.5e-15, 6, "Ion", ionCode);
    }
    
  ionCode = 1010000020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("LambdaNeutron","LambdaNeutron", 2.054, kFALSE,
		       2.5e-15, 0, "Special", ionCode);
  }

  ionCode = -1010000020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiLambdaNeutron","AntiLambdaNeutron", 2.054, kFALSE,
		       2.5e-15, 0, "Special", ionCode);
  }

  ionCode = 1020000020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("Hdibaryon","Hdibaryon", 2.23, kFALSE,
		       2.5e-15, 0, "Special", ionCode);
  }

  ionCode = -1020000020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiHdibaryon","AntiHdibaryon", 2.23, kFALSE,
		       2.5e-15, 0, "Special", ionCode);
  }

  ionCode = 1010000030;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("LambdaNeutronNeutron","LambdaNeutronNeutron", 2.99, kFALSE,
		       2.5e-15, 0, "Special", ionCode);
  }

  ionCode = -1010000030;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiLambdaNeutronNeutron","AntiLambdaNeutronNeutron", 2.99, kFALSE,
		       2.5e-15, 0, "Special", ionCode);
  }

  ionCode = 1020010020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("Xi0Proton","Xi0Proton", 2.248, kFALSE,
		       5e-15, 3, "Ion", ionCode);
  }

  ionCode = -1020010020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiXi0Proton","AntiXi0Proton", 2.248, kFALSE,
		       5e-15, 3, "Ion", ionCode);
  }

  ionCode = 1030000020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("OmegaProton","OmegaProton", 2.592, kFALSE,
		       2.5e-15, 0, "Special", ionCode);
  }

  ionCode = -1030000020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiOmegaProton","AntiOmegaProton", 2.592, kFALSE,
		       2.5e-15, 0, "Special", ionCode);
  }

  ionCode = 1030010020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("OmegaNeutron","OmegaNeutron", 2.472, kFALSE,
		       0.003, 3, "Special", ionCode);
  }

  ionCode = -1030010020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiOmegaNeutron","AntiOmegaNeutron", 2.472, kFALSE,
		       0.003, 3, "Special", ionCode);
  }

  ionCode = 1060020020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("OmegaOmega","OmegaOmega", 3.229, kFALSE,
		       2.5e-15, 6, "Special", ionCode);
  }

  ionCode = -1060020020;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiOmegaOmega","AntiOmegaOmega", 3.229, kFALSE,
		       2.5e-15, 6, "Special", ionCode);
  }

  ionCode = 1010010021;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("Lambda1405Proton","Lambda1405Proton", 2.295, kFALSE,
		       0.05, 3, "Special", ionCode);
  }

  ionCode = -1010010021;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiLambda1405Proton","AntiLambda1405Proton", 2.295, kFALSE,
		       0.05, 3, "Special", ionCode);
  }

  ionCode = 1020000021;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("Lambda1405Lambda1405","Lambda1405Lambda1405", 2.693, kFALSE,
		       0.05, 0, "Special", ionCode);
  }

  ionCode = -1020000021;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("AntiLambda1405Lambda1405","AntiLambda1405Lambda1405", 2.693, kFALSE,
		       0.05, 0, "Special", ionCode);
  }




  // Special resonances

  ionCode = 9010221;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("f0_980","f0_980", 0.980, kFALSE,
		       0.07, 0, "Resonance", ionCode);
  }

  ionCode = 225;
  if(!pdgDB->GetParticle(ionCode)){
    pdgDB->AddParticle("f2_1270","f2_1270", 1.275, kFALSE,
		       0.185, 0, "Resonance", ionCode);
  }

  // Xi-/+ (1820)
  ionCode = 123314;
  if (!pdgDB->GetParticle(ionCode)) {
    pdgDB->AddParticle("Xi_Minus_1820","Xi_Minus_1820",1.8234,kFALSE,0.024,-3,"Resonance",ionCode);
  }
  if (!pdgDB->GetParticle(-ionCode)) {
    pdgDB->AddParticle("Xi_Plus_1820","Xi_Plus_1820",1.8234,kFALSE,0.024,3,"Resonance",-ionCode);
  }

  // Xi0 (1820)
  ionCode = 123324;
  if (!pdgDB->GetParticle(ionCode)) {
    pdgDB->AddParticle("Xi_0_1820","Xi_0_1820",1.8234,kFALSE,0.024,0,"Resonance",ionCode);
  }
  if (!pdgDB->GetParticle(-ionCode)) {
    pdgDB->AddParticle("Xi_0_Bar_1820","Xi_0_Bar_1820",1.8234,kFALSE,0.024,0,"Resonance",-ionCode);
  }

  // Ps - hidden strange (s-sbar) pentaquarks

  ionCode = 9322134;
  if(!pdgDB->GetParticle(ionCode)) {
    pdgDB->AddParticle("ps_2100","ps_2100", 2.100, kFALSE,
                       0.040, 3, "Resonance", ionCode);
  }
  if(!pdgDB->GetParticle(-ionCode)) {
    pdgDB->AddParticle("anti-ps_2100","anti-ps_2100", 2.100, kFALSE,
                       0.040, -3, "Resonance", -ionCode);
  }

  ionCode = 9322136;
  if(!pdgDB->GetParticle(ionCode)) {
    pdgDB->AddParticle("ps_2500","ps_2500", 2.500, kFALSE,
                       0.040, 3, "Resonance", ionCode);
  }
  if(!pdgDB->GetParticle(-ionCode)) {
    pdgDB->AddParticle("anti-ps_2500","anti-ps_2500", 2.500, kFALSE,
                       0.040, -3, "Resonance", -ionCode);
  }

  // d*(2380) - dibaryon resonance

  ionCode = 900010020; //Arbitrary choice - as deuteron but with leading 9 instead of 10
  if(!pdgDB->GetParticle(ionCode)) {
    pdgDB->AddParticle("d*_2380","d*_2380", 2.38, kFALSE,
                       0.070, 3, "Resonance", ionCode);
  }
  pdgDB->AddAntiParticle("d*_2380_bar", - ionCode);
}
