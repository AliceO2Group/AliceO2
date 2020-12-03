/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include "TSystem.h"
#endif

Bool_t isLibrary(const char* libName)
{
  if (TString(gSystem->DynamicPathName(libName, kTRUE)) != TString(""))
    return kTRUE;
  else
    return kFALSE;
}

void g3libs()
{
  std::cout << "Loading Geant3 libraries ..." << std::endl;

  if (isLibrary("libdummies"))
    gSystem->Load("libdummies");
  // libdummies.so needed from geant3_+vmc version 0.5

  gSystem->Load("libgeant321");

  std::cout << "Loading Geant3 libraries ... finished" << std::endl;
}
