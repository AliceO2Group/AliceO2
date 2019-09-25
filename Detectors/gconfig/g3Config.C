/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// $Id: g3Config.C,v 1.1.1.1 2005/06/23 07:14:09 dbertini Exp $
//
// Configuration macro for Geant3 VirtualMC

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TGeant3.h"
#include "TGeant3TGeo.h"
#include "FairRunSim.h"
#include <iostream>
#endif
#include "commonConfig.C"

void Config()
{
  FairRunSim* run = FairRunSim::Instance();
  TString* gModel = run->GetGeoModel();
  TGeant3* geant3 = nullptr;
  if (strncmp(gModel->Data(), "TGeo", 4) == 0) {
    geant3 = new TGeant3TGeo("C++ Interface to Geant3");
    cout << "-I- G3Config: Geant3 with TGeo has been created." << endl;
  } else {
    geant3 = new TGeant3("C++ Interface to Geant3");
    cout << "-I- G3Config: Geant3 native has been created." << endl;
  }
  stackSetup(geant3, run);

  // ******* GEANT3  specific configuration for simulated Runs  *******
  geant3->SetTRIG(1); // Number of events to be processed
  geant3->SetSWIT(4, 100);
  geant3->SetDEBU(0, 0, 1);

  geant3->SetRAYL(1);
  geant3->SetSTRA(0);

  // NOTE: Please avoid changing this setting, unless justified as this might lead to very many steps
  // performed by G3; AUTO(1) is the G3 default
  geant3->SetAUTO(1); // Select automatic STMIN etc... calc. (AUTO 1) or manual (AUTO 0)

  geant3->SetABAN(0); // Restore 3.16 behaviour for abandoned tracks
  geant3->SetOPTI(2); // Select optimisation level for GEANT geometry searches (0,1,2)
  geant3->SetERAN(5.e-7);
  geant3->SetCKOV(1); // cerenkov photons

  // allow many steps per track (per volume)
  // since this is needed in the TPC
  // (this does not seem to be possible per module)
  geant3->SetMaxNStep(1E5);
}
