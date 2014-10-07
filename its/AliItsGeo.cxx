/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             * 
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *  
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#include "AliItsGeo.h"
#include "FairGeoNode.h"

ClassImp(AliItsGeo)

// -----   Default constructor   -------------------------------------------
AliItsGeo::AliItsGeo()
  : FairGeoSet()
{
  // Constructor
  // fName has to be the name used in the geometry for all volumes.
  // If there is a mismatch the geometry cannot be build.
  fName="newdetector";
  maxSectors=0;
  maxModules=10;
}

// -------------------------------------------------------------------------

const char* AliItsGeo::getModuleName(Int_t m)
{
  /** Returns the module name of AliIts number m
      Setting AliIts here means that all modules names in the
      ASCII file should start with AliIts otherwise they will
      not be constructed
  */
  sprintf(modName,"AliIts%i",m+1);
  return modName;
}

const char* AliItsGeo::getEleName(Int_t m)
{
  /** Returns the element name of Det number m */
  sprintf(eleName,"AliIts%i",m+1);
  return eleName;
}
