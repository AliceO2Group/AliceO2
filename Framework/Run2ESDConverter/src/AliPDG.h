#ifndef ALI_PDG__H
#define ALI_PDG__H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

#include "TObject.h"

// Class to encapsulate the ALICE updates to TDatabasePDG.h
// Can be used by TGeant3 and TGeant4
// Comments to: andreas.morsch@cern.ch 

class AliPDG : public TObject {
public:
    static void AddParticlesToPdgDataBase();
 private:    
    ClassDef(AliPDG,1)  // PDG database related information
};


#endif //ALI_PDG__H
