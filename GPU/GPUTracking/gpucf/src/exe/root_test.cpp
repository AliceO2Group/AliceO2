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
