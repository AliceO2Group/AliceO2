/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -----------------------------------------------------------------------------
// -----                                                                   -----
// -----                             FairSink                              -----
// -----                    Created 22.12.2017 by R. Karabowicz            -----
// -----                                                                   -----
// -----------------------------------------------------------------------------

#include "FairSink.h"

#include <TFolder.h>
#include <TTree.h>

FairSink::FairSink()
    : fRunId(0)
    , fOutFolder(0)
{}

FairSink::FairSink(const FairSink& sink)
    : fRunId(sink.fRunId)
    , fOutFolder(0)
{}

FairSink::~FairSink() {}

ClassImp(FairSink);
