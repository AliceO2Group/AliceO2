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

#ifndef FAIRSOURCE_H
#define FAIRSOURCE_H

#include "FairLogger.h"
#include "TClass.h"

#include <Rtypes.h>
#include <TObject.h>

class FairEventHeader;

enum Source_Type
{
    kONLINE,
    kFILE
};

class FairSource : public TObject
{
  public:
    FairSource();
    FairSource(const FairSource& source);
    virtual ~FairSource();
    virtual Bool_t Init() = 0;
    virtual Int_t ReadEvent(UInt_t = 0) = 0;
    virtual Bool_t SpecifyRunId() = 0;
    virtual void Close() = 0;
    virtual void Reset() = 0;
    virtual Bool_t ActivateObject(TObject**, const char*) { return kFALSE; }
    virtual Bool_t ActivateObjectAny(void**, const std::type_info&, const char*) { return kFALSE; }
    virtual Source_Type GetSourceType() = 0;
    virtual void SetParUnpackers() = 0;
    virtual Bool_t InitUnpackers() = 0;
    virtual Bool_t ReInitUnpackers() = 0;
    /**Check the maximum event number we can run to*/
    virtual Int_t CheckMaxEventNo(Int_t = 0) { return -1; }
    /**Read the tree entry on one branch**/
    virtual void ReadBranchEvent(const char*) { return; }
    virtual void ReadBranchEvent(const char*, Int_t) { return; }
    virtual void FillEventHeader(FairEventHeader* feh);
    void SetRunId(Int_t runId) { fRunId = runId; }
    Int_t GetRunId() const { return fRunId; }

  protected:
    Int_t fRunId;

  public:
    ClassDef(FairSource, 2);
};

namespace {

template<typename S>
bool ActivateObjectAnyImpl(S* source, void** obj, const std::type_info& info, const char* brname)
{
    // we check if the types match at all
    auto br = source->GetBranch(brname);
    if (!br) {
        // branch not found in source
        return false;
    }

    // look up the TClass and resulting typeid stored in this branch
    auto cl = TClass::GetClass(br->GetClassName());
    if (!cl) {
        // class not found
        return false;
    }

    auto storedtype = cl->GetTypeInfo();

    // check consistency of types
    if (info.hash_code() != storedtype->hash_code()) {
        LOG(info) << "Trying to read from branch " << brname << " with wrong type " << info.name()
                  << " (expected: " << storedtype->name() << ")\n";
        return false;
    }
    source->SetBranchStatus(brname, 1);
    // force to use the (void*) interface which is non-checking
    source->SetBranchAddress(brname, (void*)obj);
    return true;
}

}   // namespace

#endif
