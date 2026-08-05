/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Created : 11/11/03 by Ilse Koenig

///////////////////////////////////////////////////////////////////////////////
// FairGeoBuilder
//
// Base class for GEANT and ROOT builder classes to create the geometry
//
///////////////////////////////////////////////////////////////////////////////

#include "FairGeoBuilder.h"

ClassImp(FairGeoBuilder);

FairGeoBuilder::FairGeoBuilder()
    : TNamed()
    , nRot(0)
    , nMed(0)
{
    // Default constructor
}

FairGeoBuilder::FairGeoBuilder(const char* name, const char* title)
    : TNamed(name, title)
    , nRot(0)
    , nMed(0)
{
    // Constructor
}
