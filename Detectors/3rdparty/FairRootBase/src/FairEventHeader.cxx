/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                         FairEventHeader source file           -----
// -----            Created 08/09/04                 D. Bertini        -----
// -------------------------------------------------------------------------

#include "FairEventHeader.h"

#include "FairRootManager.h"

FairEventHeader::FairEventHeader()
    : fRunId(0)
    , fEventTime(-1.)
    , fInputFileId(0)
    , fMCEntryNo(0)
{}

FairEventHeader::~FairEventHeader() {}

void FairEventHeader::Register(Bool_t Persistence)
{
    FairRootManager::Instance()->Register("EventHeader.", "EvtHeader", this, Persistence);
}

ClassImp(FairEventHeader);
