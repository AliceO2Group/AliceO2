// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PWGDQCore/MCSignal.h"

ClassImp(MCSignal);

//________________________________________________________________________________________________
MCSignal::MCSignal() : TNamed("", ""),
                       fProngs({}),
                       fNProngs(0),
                       fCommonAncestorIdxs({}),
                       fTempAncestorLabel(-1)
{
}

//________________________________________________________________________________________________
MCSignal::MCSignal(int nProngs, const char* name /*= ""*/, const char* title /*= ""*/) : TNamed(name, title),
                                                                                         fProngs({}),
                                                                                         fNProngs(nProngs),
                                                                                         fCommonAncestorIdxs({}),
                                                                                         fTempAncestorLabel(-1)
{
  fProngs.reserve(nProngs);
}

//________________________________________________________________________________________________
MCSignal::MCSignal(const char* name, const char* title, std::vector<MCProng> prongs, std::vector<short> commonAncestors) : TNamed(name, title),
                                                                                                                           fProngs(prongs),
                                                                                                                           fNProngs(prongs.size()),
                                                                                                                           fCommonAncestorIdxs(commonAncestors),
                                                                                                                           fTempAncestorLabel(-1)
{
}

//________________________________________________________________________________________________
void MCSignal::SetProngs(std::vector<MCProng> prongs, std::vector<short> commonAncestors)
{
  fProngs = prongs;
  fNProngs = fProngs.size();
  fCommonAncestorIdxs = commonAncestors;
}

//________________________________________________________________________________________________
void MCSignal::AddProng(MCProng prong, short commonAncestor)
{
  if (fProngs.size() < fNProngs) {
    fProngs.push_back(prong);
    fCommonAncestorIdxs.push_back(commonAncestor);
  } else { // TODO: there should be an error message here
    return;
  }
}

//________________________________________________________________________________________________
void MCSignal::Print()
{
  cout << "Name/Title: " << fName << " / " << fTitle << endl;
  cout << "Printing " << fNProngs << "/" << fProngs.size() << " prongs:" << endl;
  int i = 0;
  for (auto& pr : fProngs) {
    cout << "Prong #" << i << "  commonAncestor" << fCommonAncestorIdxs[i] << " ================ " << endl;
    i++;
    pr.Print();
  }
}
