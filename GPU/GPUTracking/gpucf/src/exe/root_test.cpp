// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <TFile.h>
#include <TLeaf.h>
#include <TTree.h>

#include <iostream>


int main()
{
    TFile f("data/tpcdigits_ev10.root");
    f.ls();

    TTree *t = reinterpret_cast<TTree *>(f.Get("o2sim"));

    TBranch *b = t->GetBranch("TPCDigit_0");
    std::cout << b->GetClassName() << std::endl;
    std::cout << b->GetEntryNumber() << std::endl;

    b = t->GetBranch("TPCDigitMCTruth_0");
    std::cout << b->GetClassName() << std::endl;
    std::cout << b->GetEntryNumber() << std::endl;

    /* TLeaf *l = t->GetLeaf("TPCDigit_0.mTimeStamp"); */

    /* std::cout << l->GetLen() << std::endl; */
    /* std::cout << l->GetNdata() << std::endl; */

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
