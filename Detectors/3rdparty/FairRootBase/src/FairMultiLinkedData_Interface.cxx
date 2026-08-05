/*
 * FairMultiLinkedData_Interface.cxx
 *
 *  Created on: Dec 7, 2009
 *      Author: stockman
 */

#include "FairMultiLinkedData_Interface.h"

#include "FairRootManager.h"   // for FairRootManager

ClassImp(FairMultiLinkedData_Interface);

FairMultiLinkedData_Interface::FairMultiLinkedData_Interface(FairMultiLinkedData& links, Bool_t)
{
    SetLinks(links);
}

FairMultiLinkedData_Interface::FairMultiLinkedData_Interface(TString dataType,
                                                             std::vector<Int_t> links,
                                                             Int_t fileId,
                                                             Int_t evtId,
                                                             Bool_t persistanceCheck,
                                                             Bool_t bypass,
                                                             Float_t mult)
{
    FairMultiLinkedData data(dataType, links, fileId, evtId, persistanceCheck, bypass, mult);
    SetLinks(data);
}

FairMultiLinkedData_Interface::FairMultiLinkedData_Interface(Int_t dataType,
                                                             std::vector<Int_t> links,
                                                             Int_t fileId,
                                                             Int_t evtId,
                                                             Bool_t persistanceCheck,
                                                             Bool_t bypass,
                                                             Float_t mult)
{
    FairMultiLinkedData data(dataType, links, fileId, evtId, persistanceCheck, bypass, mult);
    SetLinks(data);
}

FairMultiLinkedData_Interface::FairMultiLinkedData_Interface(const FairMultiLinkedData_Interface& toCopy)
    : TObject(toCopy)
{
    if (toCopy.GetPointerToLinks() != 0) {
        SetInsertHistory(kFALSE);
        SetLinks(*(toCopy.GetPointerToLinks()));
        SetEntryNr(toCopy.GetEntryNr());
        // copy history flag information
        SetInsertHistory(toCopy.GetPointerToLinks()->GetInsertHistory());
    }
}

FairMultiLinkedData_Interface& FairMultiLinkedData_Interface::operator=(const FairMultiLinkedData_Interface& rhs)
{
    if (rhs.GetPointerToLinks() != 0) {
        SetInsertHistory(kFALSE);
        SetLinks(*(rhs.GetPointerToLinks()));
        SetEntryNr(rhs.GetEntryNr());
        SetInsertHistory(kTRUE);
    }
    return *this;
}

FairMultiLinkedData* FairMultiLinkedData_Interface::CreateFairMultiLinkedData()
{
    if (FairRootManager::Instance() != nullptr) {
        if (FairRootManager::Instance()->GetUseFairLinks()) {
            if (fLink == nullptr) {
                fLink.reset(new FairMultiLinkedData());
            }
            return fLink.get();
        }
    }
    return 0;
}

std::set<FairLink> FairMultiLinkedData_Interface::GetLinks() const
{
    if (GetPointerToLinks() != 0) {
        return GetPointerToLinks()->GetLinks();
    } else {
        std::set<FairLink> emptySet;
        return emptySet;
    }
}

Int_t FairMultiLinkedData_Interface::GetNLinks() const
{
    if (GetPointerToLinks() != 0) {
        return GetPointerToLinks()->GetNLinks();
    } else {
        return 0;
    }
}

FairLink FairMultiLinkedData_Interface::GetLink(Int_t pos) const
{
    if (GetPointerToLinks() != 0) {
        return GetPointerToLinks()->GetLink(pos);
    } else {
        FairLink emptyLink;
        return emptyLink;
    }
}

FairLink FairMultiLinkedData_Interface::GetEntryNr() const
{
    if (GetPointerToLinks() != 0) {
        return GetPointerToLinks()->GetEntryNr();
    } else {
        FairLink emptyLink;
        return emptyLink;
    }
}

FairMultiLinkedData FairMultiLinkedData_Interface::GetLinksWithType(Int_t type) const
{
    if (GetPointerToLinks() != 0) {
        return GetPointerToLinks()->GetLinksWithType(type);
    } else {
        FairMultiLinkedData emptyLinks;
        return emptyLinks;
    }
}

std::vector<FairLink> FairMultiLinkedData_Interface::GetSortedMCTracks()
{
    if (GetPointerToLinks() != 0) {
        return GetPointerToLinks()->GetSortedMCTracks();
    } else {
        std::vector<FairLink> empty;
        return empty;
    }
}

void FairMultiLinkedData_Interface::SetLinks(FairMultiLinkedData links)
{
    CreateFairMultiLinkedData();
    if (GetPointerToLinks() != 0) {
        GetPointerToLinks()->SetLinks(links);
    }
}

void FairMultiLinkedData_Interface::SetLink(FairLink link)
{
    CreateFairMultiLinkedData();
    if (GetPointerToLinks() != 0) {
        GetPointerToLinks()->SetLink(link);
    }
}

void FairMultiLinkedData_Interface::SetEntryNr(FairLink val)
{
    CreateFairMultiLinkedData();
    if (GetPointerToLinks() != 0) {
        GetPointerToLinks()->SetEntryNr(val);
    }
}

void FairMultiLinkedData_Interface::AddLinks(FairMultiLinkedData links, Float_t mult)
{
    CreateFairMultiLinkedData();
    if (GetPointerToLinks() != 0) {
        GetPointerToLinks()->AddLinks(links, mult);
    }
}

void FairMultiLinkedData_Interface::AddLink(FairLink link)
{
    CreateFairMultiLinkedData();
    if (GetPointerToLinks() != 0) {
        GetPointerToLinks()->AddLink(link);
    }
}

void FairMultiLinkedData_Interface::AddInterfaceData(FairMultiLinkedData_Interface* data)
{
    SetInsertHistory(kFALSE);   // todo add previous history value
    if (data->GetEntryNr().GetType() != -1)
        AddLink(data->GetEntryNr());
    else
        std::cout << "-E- FairMultiLinkedData_Interface::AddInterfaceData EntryNr == " << data->GetEntryNr()
                  << std::endl;

    if (data->GetPointerToLinks() != 0) {
        AddLinks(*data->GetPointerToLinks());
    }
    SetInsertHistory(GetPointerToLinks()->GetInsertHistory());
}

void FairMultiLinkedData_Interface::SetInsertHistory(Bool_t val)
{
    CreateFairMultiLinkedData();
    if (GetPointerToLinks() != 0) {
        GetPointerToLinks()->SetInsertHistory(val);
    }
}

void FairMultiLinkedData_Interface::ResetLinks()
{
    if (GetPointerToLinks() != 0) {
        GetPointerToLinks()->ResetLinks();
    }
}
