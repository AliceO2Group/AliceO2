/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/*
 * FairLink.cpp
 *
 *  Created on: Dec 23, 2009
 *      Author: stockman
 */

#include "FairLink.h"

#include "FairRootManager.h"

ClassImp(FairLink);

FairLink::FairLink(TString branchName, Int_t index, Float_t weight)
    : fFile(0)
    , fType(0)
    , fEntry(-1)
    , fIndex(index)
    , fWeight(weight)
{
    SetType(FairRootManager::Instance()->GetBranchId(branchName));
}

FairLink::FairLink(Int_t file, Int_t entry, TString branchName, Int_t index, Float_t weight)
    : fFile(0)
    , fType(0)
    , fEntry(entry)
    , fIndex(index)
    , fWeight(weight)
{
    SetFile(file);
    SetType(FairRootManager::Instance()->GetBranchId(branchName));
}

void FairLink::PrintLinkInfo(std::ostream& out) const
{
    out << "(" << GetFile() << "/" << GetEntry() << "/";
    out << GetType() << "/" << GetIndex() << "/" << GetWeight() << ")";
}
