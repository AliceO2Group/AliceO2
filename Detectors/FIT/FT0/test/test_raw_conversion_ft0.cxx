// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TSystem.h>
#include <TTree.h>
#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE Test FT0RAWIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DetectorsCommonDataFormats/NameConf.h"
#include "Framework/Logger.h"
#include "DataFormatsFT0/Digit.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <cstring>

using namespace o2::ft0;

BOOST_AUTO_TEST_CASE(RAWTest)
{
  gSystem->Exec("$O2_ROOT/bin/o2-sim -n 10 -m FT0 -g pythia8");
  gSystem->Exec("$O2_ROOT/bin/o2-sim-digitizer-workflow -b");
  TFile flIn("ft0digits.root");
  std::unique_ptr<TTree> tree((TTree*)flIn.Get("o2sim"));
  std::vector<o2::ft0::Digit> digitsBC, *ft0BCDataPtr = &digitsBC;
  std::vector<o2::ft0::ChannelData> digitsCh, *ft0ChDataPtr = &digitsCh;
  tree->Print();
  BOOST_REQUIRE(tree);
  tree->SetBranchAddress("FT0DIGITSBC", &ft0BCDataPtr);
  tree->SetBranchAddress("FT0DIGITSCH", &ft0ChDataPtr);
  int nbc = 0, nbc2 = 0, nch = 0, nch2 = 0;
  for (int ient = 0; ient < tree->GetEntries(); ient++) {
    tree->GetEntry(ient);
    nbc = digitsBC.size();
    for (int ibc = 0; ibc < nbc; ibc++) {
      auto& bcd = digitsBC[ibc];
      int bc = bcd.getBC();
      auto channels = bcd.getBunchChannelData(digitsCh);
      nch += channels.size();
    }
  }
  std::cout<<" @@@ sim nbc "<<nbc<<" nchannels "<<nch<<std::endl;
  gSystem->Exec("$O2_ROOT/bin/o2-ft0-digi2raw --file-per-link");
  gSystem->Exec("$O2_ROOT/bin/o2-raw-file-reader-workflow -b --input-conf FT0raw.cfg|$O2_ROOT/bin/o2-ft0-flp-dpl-workflow -b");
  TFile flIn2("o2digit_ft0.root");
  std::unique_ptr<TTree> tree2((TTree*)flIn2.Get("o2sim"));
  std::vector<o2::ft0::Digit> digitsBC2, *ft0BCDataPtr2 = &digitsBC2;
  std::vector<o2::ft0::ChannelData> digitsCh2, *ft0ChDataPtr2 = &digitsCh2;
  BOOST_REQUIRE(tree2);
  tree2->Print();
  tree2->SetBranchAddress("FT0DIGITSBC", &ft0BCDataPtr2);
  tree2->SetBranchAddress("FT0DIGITSCH", &ft0ChDataPtr2);
  for (int ient = 0; ient < tree2->GetEntries(); ient++) {
    tree2->GetEntry(ient);
    nbc2 = digitsBC2.size();
    for (int ibc = 0; ibc < nbc2; ibc++) {
      auto& bcd2 = digitsBC2[ibc];
      int bc2 = bcd2.getBC();
      auto channels2 = bcd2.getBunchChannelData(digitsCh2);
      nch2 += channels2.size();
    }
  }
  std::cout<<" @@@ reco nbc "<<nbc2<<" nchannels "<<nch2<<std::endl;
  std::cout << " comparison rec " << nbc2 << " " << nch2 << " sim " << nbc << " " << nch << std::endl;

  BOOST_CHECK(nbc == nbc2);
  BOOST_CHECK(nch == nch2);
};
