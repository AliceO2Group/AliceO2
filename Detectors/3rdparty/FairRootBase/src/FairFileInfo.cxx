/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//  FairFileInfo.cxx
//  Created by Mohammad Al-Turany on 6/21/11.
//
#include "FairFileInfo.h"

#include <TFile.h>     // for TFile
#include <TSystem.h>   // for TSystem, gSystem
#include <stdio.h>     // for printf

//__________________________________________________________________________
FairFileInfo::FairFileInfo()
    : TNamed()
    , fPath("")
    , fSize(0)
    , fIdentifier(0)
    , fInChainId(0)
{}
//__________________________________________________________________________
FairFileInfo::FairFileInfo(TFile* file, UInt_t id, UInt_t ChId)
    : TNamed(file->GetName(), "Inputfile")
    , fPath("")
    , fSize(file->GetSize())
    , fIdentifier(id)
    , fInChainId(ChId)
{
    fPath += gSystem->WorkingDirectory();
    fPath += fName;
}
//__________________________________________________________________________
FairFileInfo::~FairFileInfo() {}
//__________________________________________________________________________
void FairFileInfo::Print(Option_t*) const
{
    printf("Printing FairFileInfo for    :  %s \n", fName.Data());
    printf("File full path               :  %s \n", fPath.Data());
    printf("File Size in Byte            :  %i \n", fSize);
    printf("File Identifier              :  %i \n", fIdentifier);
    printf("File order in input chain    :  %i \n", fInChainId);
}
//__________________________________________________________________________
ClassImp(FairFileInfo);
