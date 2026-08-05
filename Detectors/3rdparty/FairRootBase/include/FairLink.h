/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/*
 * FairLink.h
 *
 *  Created on: Dec 23, 2009
 *      Author: stockman
 *      "Pointer" to any data structure inside a root file with a tree structure
 *      Contains four information:
 *      	File : in which file is the data written (-1 : actual number; 254 : maximum number)
 *      	Entry: entry number in branch of a tree (-1 : actual entry)
 *      	Type : branch inside the tree (-1 : FairLink not set; 65534 : maximum number)
 *      	Index: position inside a container stored in the given branch and entry
 *      	Weight: weighting factor to describe probabilities of a MC truth match
 *
 */

#ifndef FAIRLINK_H_
#define FAIRLINK_H_

#include <Rtypes.h>   // for Int_t, Float_t, etc, ClassDefNV
#include <iostream>   // for ostream, cout

class FairLink
{
  public:
    FairLink();
    FairLink(Int_t type, Int_t index, Float_t weight = 1.);
    FairLink(TString branchName, Int_t index, Float_t weight = 1.);
    FairLink(Int_t file, Int_t entry, Int_t type, Int_t index, Float_t weight = 1.);
    FairLink(Int_t file, Int_t entry, TString branchName, Int_t index, Float_t weight = 1.);
    ~FairLink(){};

    void SetLink(Int_t file, Int_t entry, Int_t type, Int_t index, Float_t weight = 1.)
    {
        SetFile(file);
        SetEntry(entry);
        SetType(type);
        SetIndex(index);
        SetWeight(weight);
    };
    void SetLink(Int_t type, Int_t index, Float_t weight = 1.)
    {
        SetFile(-1);
        SetEntry(-1);
        SetType(type);
        SetIndex(index);
        SetWeight(weight);
    };
    int GetFile() const;
    int GetEntry() const { return fEntry; }
    int GetType() const;
    int GetIndex() const { return fIndex; }
    float GetWeight() const { return fWeight; }

    void SetFile(int file);
    void SetEntry(int entry) { fEntry = entry; };
    void SetType(int type);
    void SetIndex(int index) { fIndex = index; };

    void SetWeight(Float_t weight) { fWeight = weight; }
    void AddWeight(Float_t weight) { fWeight += weight; }

    void PrintLinkInfo(std::ostream& out = std::cout) const;

    bool operator==(const FairLink& link) const
    {
        if ((GetFile() == link.GetFile() || link.GetFile() == -1)
            && (GetEntry() == link.GetEntry() || link.GetEntry() == -1) && GetType() == link.GetType()
            && GetIndex() == link.GetIndex()) {
            return true;
        } else {
            return false;
        }
    }

    bool operator<(const FairLink& link) const
    {
        if (GetFile() != -1 && link.GetFile() != -1) {
            if (GetFile() < link.GetFile())
                return true;
            else if (link.GetFile() < GetFile())
                return false;
        }
        if (GetEntry() != -1 && link.GetEntry() != -1) {
            if (GetEntry() < link.GetEntry())
                return true;
            else if (link.GetEntry() < GetEntry())
                return false;
        }
        if (GetType() < link.GetType())
            return true;
        else if (link.GetType() < GetType())
            return false;
        if (GetIndex() < link.GetIndex())
            return true;
        else if (link.GetIndex() < GetIndex())
            return false;

        return false;
    }

    friend std::ostream& operator<<(std::ostream& out, const FairLink& link)
    {
        link.PrintLinkInfo(out);
        return out;
    }

    ClassDefNV(FairLink, 5);

    template<class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar& fFile;
        ar& fType;
        ar& fEntry;
        ar& fIndex;
        ar& fWeight;
    }

  private:
    unsigned char fFile;
    unsigned short fType;
    int fEntry;
    int fIndex;
    float fWeight;
};

inline FairLink::FairLink()
    : fFile(0)
    , fType(0)
    , fEntry(-1)
    , fIndex(-1)
    , fWeight(1.0)
{}

inline FairLink::FairLink(Int_t type, Int_t index, Float_t weight)
    : fFile(0)
    , fType(0)
    , fEntry(-1)
    , fIndex(index)
    , fWeight(weight)
{
    SetType(type);
}

inline FairLink::FairLink(Int_t file, Int_t entry, Int_t type, Int_t index, Float_t weight)
    : fFile(0)
    , fType(0)
    , fEntry(entry)
    , fIndex(index)
    , fWeight(weight)
{
    SetFile(file);
    SetType(type);
}

inline void FairLink::SetType(int type)
{
    if (type < -1)
        return;
    fType = type + 1;
}

inline int FairLink::GetType() const
{
    int type = fType;
    return type - 1;
}

inline void FairLink::SetFile(int file)
{
    if (file < -1)
        return;
    fFile = file + 1;
}

inline int FairLink::GetFile() const
{
    int file = fFile;
    return file - 1;
}

#endif /* FAIRLINK_H_ */
