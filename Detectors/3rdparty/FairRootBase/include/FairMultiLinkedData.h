/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/**
 * FairMultiLinkedData.h
 *
 * \date Dec 7, 2009
 * \author T.Stockmanns <t.stockmanns@fz-juelich.de>
 */

#ifndef FAIRMULTILINKEDDATA_H_
#define FAIRMULTILINKEDDATA_H_

#include "FairLink.h"   // for FairLink

#include <Rtypes.h>    // for Int_t, Bool_t, kFALSE, etc
#include <TObject.h>   // for TObject
#include <TString.h>   // for TString
#include <iosfwd>      // for ostream
#include <iostream>    // for operator<<, ostream, cout
#include <set>         // for set
#include <vector>      // for vector

class FairMultiLinkedData : public TObject
{
  public:
    FairMultiLinkedData();                                                                   ///< Default constructor
    FairMultiLinkedData(const std::set<FairLink>& links, Bool_t persistanceCheck = kTRUE);   ///< Constructor
    FairMultiLinkedData(TString dataType,
                        const std::vector<Int_t>& links,
                        Int_t fileId = -1,
                        Int_t evtId = -1,
                        Bool_t persistanceCheck = kTRUE,
                        Bool_t bypass = kFALSE,
                        Float_t mult = 1.0);   ///< Constructor
    FairMultiLinkedData(Int_t dataType,
                        const std::vector<Int_t>& links,
                        Int_t fileId = -1,
                        Int_t evtId = -1,
                        Bool_t persistanceCheck = kTRUE,
                        Bool_t bypass = kFALSE,
                        Float_t mult = 1.0);   ///< Constructor

    virtual ~FairMultiLinkedData(){};

    virtual std::set<FairLink> GetLinks() const { return fLinks; }   ///< returns stored links as FairLinks
    virtual FairLink GetEntryNr() const { return fEntryNr; }         ///< gives back the entryNr
    virtual Int_t GetNLinks() const { return fLinks.size(); }        ///< returns the number of stored links
    virtual FairLink GetLink(Int_t pos) const;                       ///< returns the FairLink at the given position
    virtual FairMultiLinkedData GetLinksWithType(
        Int_t type) const;   ///< Gives you a list of links which contain the given type
    virtual std::vector<FairLink>
        GetSortedMCTracks();   ///< Gives you a list of all FairLinks pointing to a "MCTrack" sorted by their weight
    TObject* GetData(FairLink& myLink);   ///< Get the TObject the Link is pointing to
    virtual Int_t GetDefaultType() { return fDefaultType; }
    Bool_t GetPersistanceCheck() { return fPersistanceCheck; }   ///< Returns the value of PersistanceCheck
    Int_t GetVerbose() { return fVerbose; }                      ///< Returns the verbosity level
    virtual void SetDefaultType(Int_t type) { fDefaultType = type; }
    virtual void SetPersistanceCheck(Bool_t check)
    {
        fPersistanceCheck = check;
    }   ///< Controls if a persistance check of a link is done or not
    virtual void SetVerbose(Int_t level) { fVerbose = level; }   ///< Sets the verbosity level
    virtual void SetInsertHistory(Bool_t val)
    {
        fInsertHistory = val;
    }   ///< Toggles if history of a link is inserted or not
    Bool_t GetInsertHistory() const { return fInsertHistory; }

    virtual void SetEntryNr(FairLink entry) { fEntryNr = entry; }
    virtual void SetLinks(FairMultiLinkedData links, Float_t mult = 1.0);   ///< Sets the links as vector of FairLink
    virtual void SetLink(FairLink link,
                         Bool_t bypass = kFALSE,
                         Float_t mult = 1.0);   ///< Sets the Links with a single FairLink

    virtual void AddLinks(FairMultiLinkedData links,
                          Float_t mult = 1.0);   ///< Adds a List of FairLinks (FairMultiLinkedData) to fLinks
    virtual void AddLink(FairLink link,
                         Bool_t bypass = kFALSE,
                         Float_t mult = 1.0);   ///< Adds a FairLink link at the end of fLinks. If multi is kTRUE a link
                                                ///< is allowed more than once otherwise it is stored only once

    virtual void InsertLink(FairLink link);   ///< Inserts a link into the list of links without persistance checking
    virtual void InsertHistory(
        FairLink link);   ///< Adds the FairLinks of the inserted link to the set of links of this object

    virtual void AddAllWeights(Double_t weight);        ///< Adds weight to all Links
    virtual void SetAllWeights(Double_t weight);        ///< Sets a common weight for Links
    virtual void MultiplyAllWeights(Double_t weight);   ///< Multiplies all Links with weight

    virtual Bool_t IsLinkInList(FairLink link)
    {
        return IsLinkInList(link.GetType(), link.GetIndex());
    }   ///< Test if a given link is in fLinks
    virtual Bool_t IsLinkInList(
        Int_t type,
        Int_t index);   ///< Same as IsLinkInList(FairLink) just with type and index given separately
    virtual Int_t LinkPosInList(FairLink link)
    {
        return LinkPosInList(link.GetType(), link.GetIndex());
    }   ///< returns position of link in fLinks. If it is not in the list -1 is returned
    virtual Int_t LinkPosInList(Int_t type, Int_t index);   ///< Same as LinkPosInList(FairLink)

    virtual void DeleteLink(FairLink link)
    {
        DeleteLink(link.GetType(), link.GetIndex());
    }                                                   ///< Deletes a link ouf of fLinks
    virtual void DeleteLink(Int_t type, Int_t index);   ///< Deletes a link ouf of fLinks

    virtual void Reset() { ResetLinks(); }
    virtual void ResetLinks() { fLinks.clear(); }   ///< Clears fLinks

    std::ostream& PrintLinkInfo(std::ostream& out = std::cout) const
    {
        out << GetEntryNr() << " -> [";
        for (Int_t i = 0; i < GetNLinks(); i++) {
            GetLink(i).PrintLinkInfo(out);
            out << " ";
        }
        out << "]";
        return out;
    }   ///< Output

    friend std::ostream& operator<<(std::ostream& out, const FairMultiLinkedData& data)
    {
        data.PrintLinkInfo(out);
        return out;
    }   ///< Output

  protected:
    std::set<FairLink> fLinks;
    FairLink fEntryNr;
    Bool_t fPersistanceCheck;   //!
    Bool_t fInsertHistory;      //!
    Int_t fVerbose;             //!

    virtual void SimpleAddLinks(Int_t fileId,
                                Int_t evtId,
                                Int_t dataType,
                                const std::vector<Int_t>& links,
                                Bool_t,
                                Float_t)
    {
        for (UInt_t i = 0; i < links.size(); i++) {
            fLinks.insert(FairLink(fileId, evtId, dataType, links[i]));
        }
    }
    Int_t fDefaultType;

    ClassDef(FairMultiLinkedData, 4);
};

/**\fn virtual void FairMultiLinkedData::SetLinks(Int_t type, std::vector<Int_t> links)
 * \param type as Int_t gives one type of source data for all indices
 * \param links as vector<Int_t> gives the vector of indices at which position in the TClonesArray the source data is
 * stored
 */

/**\fn virtual void FairMultiLinkedData::SetLinks(std::vector<Int_t> type, std::vector<Int_t> links)
 * \param type as vector<Int_t> gives the type of source data (TClonesArray)
 * \param links as vector<Int_t> gives the vector of indices at which position in the TClonesArray the source data is
 * stored
 */

#endif /* FAIRMULTILinkedData_H_ */
