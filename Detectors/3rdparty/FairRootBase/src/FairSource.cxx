/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -----------------------------------------------------------------------------
// -----                                                                   -----
// -----                             FairSource                            -----
// -----                    Created 01.11.2013 by F. Uhlig                 -----
// -----                                                                   -----
// -----------------------------------------------------------------------------

#include "FairSource.h"

#include "FairEventHeader.h"

FairSource::FairSource()
    : TObject()
    , fRunId(0)
{}

FairSource::FairSource(const FairSource& source)
    : TObject(source)
    , fRunId(source.fRunId)
{}

FairSource::~FairSource() {}

void FairSource::FillEventHeader(FairEventHeader* eh)
{
    if (eh) {
        eh->SetRunId(fRunId);
    }
}

ClassImp(FairSource);
