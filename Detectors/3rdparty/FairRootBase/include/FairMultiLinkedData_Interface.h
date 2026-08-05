/**
 * FairMultiLinkedData_Interface.h
 *
 * \date Dec 7, 2009
 * \author T.Stockmanns <t.stockmanns@fz-juelich.de>
 */

#ifndef FairMultiLinkedData_Interface_H_
#define FairMultiLinkedData_Interface_H_

#include "FairLink.h"   // for FairLink
#include "FairMultiLinkedData.h"

#include <Rtypes.h>    // for Int_t, Bool_t, kFALSE, etc
#include <TObject.h>   // for TObject
#include <TString.h>   // for TString
#include <iostream>    // for operator<<, ostream, cout
#include <memory>      // for std::unique_ptr
#include <set>         // for set
#include <vector>      // for vector

class FairMultiLinkedData_Interface : public TObject
{
  public:
    FairMultiLinkedData_Interface() = default;   ///< Default constructor
    FairMultiLinkedData_Interface(FairMultiLinkedData& links, Bool_t persistanceCheck = kTRUE);   ///< Constructor
    FairMultiLinkedData_Interface(TString dataType,
                                  std::vector<Int_t> links,
                                  Int_t fileId = -1,
                                  Int_t evtId = -1,
                                  Bool_t persistanceCheck = kTRUE,
                                  Bool_t bypass = kFALSE,
                                  Float_t mult = 1.0);   ///< Constructor
    FairMultiLinkedData_Interface(Int_t dataType,
                                  std::vector<Int_t> links,
                                  Int_t fileId = -1,
                                  Int_t evtId = -1,
                                  Bool_t persistanceCheck = kTRUE,
                                  Bool_t bypass = kFALSE,
                                  Float_t mult = 1.0);   ///< Constructor
    FairMultiLinkedData_Interface(const FairMultiLinkedData_Interface& toCopy);

    ~FairMultiLinkedData_Interface() = default;

    FairMultiLinkedData_Interface& operator=(const FairMultiLinkedData_Interface& rhs);

    std::set<FairLink> GetLinks() const;                      ///< returns stored links as FairLinks
    Int_t GetNLinks() const;                                  ///< returns the number of stored links
    FairLink GetLink(Int_t pos) const;                        ///< returns the FairLink at the given position
    FairMultiLinkedData GetLinksWithType(Int_t type) const;   ///< returns all FairLinks with the corresponding type
    FairLink GetEntryNr() const;
    FairMultiLinkedData* GetPointerToLinks() const { return fLink.get(); }   ///< returns non-owning pointer

    std::vector<FairLink> GetSortedMCTracks();

    void SetLinks(FairMultiLinkedData links);   ///< Sets the links as vector of FairLink
    void SetLink(FairLink link);                ///< Sets the Links with a single FairLink
    void SetInsertHistory(Bool_t val);
    void SetEntryNr(FairLink val);
    ///< @param[in] links transfers ownership
    void SetPointerToLinks(FairMultiLinkedData* links) { fLink.reset(links); }
    void SetPointerToLinks(std::unique_ptr<FairMultiLinkedData> links) { fLink = std::move(links); }

    void AddLinks(FairMultiLinkedData links,
                  Float_t mult = 1.0);   ///< Adds a List of FairLinks (FairMultiLinkedData_Interface) to fLinks
    void AddLink(FairLink link);   ///< Adds a FairLink link at the end of fLinks. If multi is kTRUE a link is allowed
                                   ///< more than once otherwise
    void AddInterfaceData(FairMultiLinkedData_Interface* data);

    void ResetLinks();

    std::ostream& PrintLinkInfo(std::ostream& out = std::cout) const
    {
        if (GetPointerToLinks() != 0)
            GetPointerToLinks()->PrintLinkInfo(out);
        return out;
    }   ///< Output

    friend std::ostream& operator<<(std::ostream& out, const FairMultiLinkedData_Interface& data)
    {
        data.PrintLinkInfo(out);
        return out;
    }   ///< Output

  protected:
    std::unique_ptr<FairMultiLinkedData> fLink;
    FairMultiLinkedData* CreateFairMultiLinkedData();   ///< returns non-owning pointer

    ClassDef(FairMultiLinkedData_Interface, 6);
};

/**\fn virtual void FairMultiLinkedData_Interface::SetLinks(Int_t type, std::vector<Int_t> links)
 * \param type as Int_t gives one type of source data for all indices
 * \param links as vector<Int_t> gives the vector of indices at which position in the TClonesArray the source data is
 * stored
 */

/**\fn virtual void FairMultiLinkedData_Interface::SetLinks(std::vector<Int_t> type, std::vector<Int_t> links)
 * \param type as vector<Int_t> gives the type of source data (TClonesArray)
 * \param links as vector<Int_t> gives the vector of indices at which position in the TClonesArray the source data is
 * stored
 */

#endif /* FairMultiLinkedData_Interface_H_ */
