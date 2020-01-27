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

#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#include "TOFReconstruction/Encoder.h"

#endif

// example of TOF raw data encoding from digits

void run_digi2raw_tof(std::string outName = "rawtof.bin",     // name of the output binary file
                      std::string inpName = "tofdigits.root", // name of the input TOF digits
                      int verbosity = 0,                      // set verbosity
                      int cache = 1024 * 1024)                // memory caching in Byte
{
  TFile* f = new TFile(inpName.c_str());
  TTree* t = (TTree*)f->Get("o2sim");

  std::vector<o2::tof::Digit> digits, *pDigits = &digits;
  std::vector<o2::tof::ReadoutWindowData> row, *pRow = &row;

  t->SetBranchAddress("TOFDigit", &pDigits);
  t->SetBranchAddress("TOFReadoutWindow", &pRow);
  t->GetEvent(0);

  int nwindow = row.size();
  int ndigits = digits.size();

  printf("Encoding %d tof window with %d digits\n", nwindow, ndigits);

  int nwindowperorbit = o2::tof::Geo::NWINDOW_IN_ORBIT;
  int nwindowintimeframe = 256 * nwindowperorbit;

  o2::tof::raw::Encoder encoder;
  encoder.setVerbose(verbosity);

  encoder.open(outName.c_str());
  encoder.alloc(cache);

  std::vector<o2::tof::Digit> digitRO;
  std::vector<o2::tof::Digit> emptyWindow;
  std::vector<std::vector<o2::tof::Digit>> digitWindows;

  for (int i = 0; i < nwindow; i += nwindowperorbit) {
    digitWindows.clear();

    // push all windows in the current orbit in the structure
    for (int j = i; j < i + nwindowperorbit; j++) {
      if (j < nwindow) {
        digitRO.clear();
        for (int id = 0; id < row.at(j).size(); id++)
          digitRO.push_back(digits[row.at(j).first() + id]);
        digitWindows.push_back(digitRO);
      } else {
        digitWindows.push_back(emptyWindow);
      }
    }

    if (verbosity)
      printf("----------\nwindow = %d\n----------\n", i);
    encoder.encode(digitWindows, i);
  }

  encoder.flush();
  encoder.close();
}
