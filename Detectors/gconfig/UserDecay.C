/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include "TDatabasePDG.h"
#include "TParticlePDG.h"
#include "TVirtualMC.h"
#endif

void UserDecayConfig()
{
  cout << "Loading User Decay Config from macro" << endl;
  TDatabasePDG* db = TDatabasePDG::Instance();
  TParticlePDG* p = 0;

  Int_t mode[6][3];
  Float_t bratio[6];
  Int_t AlphaPDG, He5PDG;
  p = db->GetParticle("Alpha");
  if (p)
    AlphaPDG = p->PdgCode();
  p = db->GetParticle("He5");

  if (p)
    He5PDG = p->PdgCode();
  for (Int_t kz = 0; kz < 6; kz++) {
    bratio[kz] = 0.;
    mode[kz][0] = 0;
    mode[kz][1] = 0;
    mode[kz][2] = 0;
    //  cout << mode[kz][0] << " " << 	mode[kz][1] << " " << mode[kz][2] << endl;
  }
  bratio[0] = 100.;
  mode[0][0] = 2112;
  mode[0][1] = AlphaPDG;

  /*  bratio[1] = 50.;
   mode[1][0] =2212  ;
   mode[1][1] =AlphaPDG  ;
    
  */
  TVirtualMC::GetMC()->SetDecayMode(He5PDG, bratio, mode);
}
