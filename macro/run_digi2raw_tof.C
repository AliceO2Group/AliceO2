// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TTree.h>
#include <TFile.h>
#include <vector>
#include <string>

#include "TOFBase/Digit.h"
#include "TOFReconstruction/Encoder.h"

#endif

// example of TOF raw data encoding from digits

void run_digi2raw_tof(std::string outName = "rawtof.bin",     // name of the output binary file
                      std::string inpName = "tofdigits.root", // name of the input TOF digits
                      int verbosity = 0,                      // set verbosity
                      int cache = 100000000)                  // memory caching in Byte
{
  TFile* f = new TFile(inpName.c_str());
  TTree* t = (TTree*)f->Get("o2sim");

  std::vector<std::vector<o2::tof::Digit>> digits, *pDigits = &digits;

  t->SetBranchAddress("TOFDigit", &pDigits);
  t->GetEvent(0);

  int nwindow = digits.size();

  printf("Encoding %d tof window\n", nwindow);

  o2::tof::compressed::Encoder encoder;
  encoder.setVerbose(verbosity);

  encoder.open(outName.c_str());
  encoder.alloc(cache);

  for (int i = 0; i < nwindow; i++) {
    if (verbosity)
      printf("----------\nwindow = %d\n----------\n", i);
    encoder.encode(digits.at(i), i);
  }

  encoder.flush();
  encoder.close();
}
