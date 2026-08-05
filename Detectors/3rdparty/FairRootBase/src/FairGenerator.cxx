/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                     FairGenerator source file                 -----
// -------------------------------------------------------------------------

#include "FairGenerator.h"

FairGenerator::FairGenerator()
    : TNamed()
{}

FairGenerator::FairGenerator(const char* name, const char* title)
    : TNamed(name, title)
{}

FairGenerator::FairGenerator(const FairGenerator& rhs)
    : TNamed(rhs)
{}

FairGenerator::~FairGenerator() {}

FairGenerator& FairGenerator::operator=(const FairGenerator& rhs)
{
    // check assignment to self
    if (this == &rhs)
        return *this;

    // base class assignment
    TNamed::operator=(rhs);

    return *this;
}

FairGenerator* FairGenerator::CloneGenerator() const
{
    Fatal("CloneGenerator", "Has to be overriden in multi-threading applications.");
    return 0;
}

ClassImp(FairGenerator);
