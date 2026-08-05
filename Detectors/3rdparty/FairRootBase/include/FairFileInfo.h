/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//  FairFileInfo.h
//  Created by Mohammad Al-Turany on 6/21/11.
//
#ifndef FAIRFILEINFO_H
#define FAIRFILEINFO_H

#include <Rtypes.h>    // for UInt_t, FairFileInfo::Class, etc
#include <TNamed.h>    // for TNamed
#include <TString.h>   // for TString

class TFile;

class FairFileInfo : public TNamed
{
  public:
    FairFileInfo();
    FairFileInfo(TFile* file, UInt_t id, UInt_t ChId);
    virtual ~FairFileInfo();

    TString GetPath() { return fPath; }
    UInt_t GetSize() { return fSize; }
    UInt_t GetIdentifier() { return fIdentifier; }
    UInt_t GetOrderInChain() { return fInChainId; }

    void Print(Option_t* option = "") const;

    void SetPath(TString path) { fPath = path; }
    void SetSize(UInt_t size) { fSize = size; }
    void SetIdentifier(UInt_t id) { fIdentifier = id; }
    void SetOrderInChain(UInt_t id) { fInChainId = id; }

  protected:
    /** Full path of file*/
    TString fPath;
    /** size of file in bytes*/
    UInt_t fSize;
    /** Signal file identifier used*/
    UInt_t fIdentifier;
    /** file order in the chain*/
    UInt_t fInChainId;

  private:
    FairFileInfo(const FairFileInfo&);
    FairFileInfo& operator=(const FairFileInfo&);

    ClassDef(FairFileInfo, 1);
};

#endif   // FAIRFILEINFO_H
