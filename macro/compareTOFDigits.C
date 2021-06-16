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

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TTree.h>
#include <TFile.h>
#include <vector>
#include <string>

#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"

#endif

bool compareTOFDigits(std::string inpName1 = "tofdigitsOr.root", std::string inpName2 = "tofdigits.root")
{
  bool status = true;
  int ngood = 0;
  int nfake = 0;

  TFile* f1 = TFile::Open(inpName1.c_str());
  TFile* f2 = TFile::Open(inpName2.c_str());

  TTree* t1 = (TTree*)f1->Get("o2sim");
  TTree* t2 = (TTree*)f2->Get("o2sim");

  std::vector<o2::tof::Digit> digits1, *pDigits1 = &digits1;
  std::vector<o2::tof::ReadoutWindowData> row1, *pRow1 = &row1;
  t1->SetBranchAddress("TOFDigit", &pDigits1);
  t1->SetBranchAddress("TOFReadoutWindow", &pRow1);
  std::vector<o2::tof::Digit> digits2, *pDigits2 = &digits2;
  std::vector<o2::tof::ReadoutWindowData> row2, *pRow2 = &row2;
  t2->SetBranchAddress("TOFDigit", &pDigits2);
  t2->SetBranchAddress("TOFReadoutWindow", &pRow2);

  t1->GetEvent(0);
  t2->GetEvent(0);

  int nro1 = row1.size();
  int nro2 = row2.size();

  for (int ii = 1; ii < t2->GetEntries(); ii++) {
    t2->GetEvent(ii);
    nro2 += row2.size();
  }

  int row2lastSize = row2.size();

  while (row1[nro1 - 1].size() == 0 && nro1 > 0)
    nro1--;
  while (row2[row2lastSize - 1].size() == 0 && row2lastSize > 0) {
    row2lastSize--;
    nro2--;
  }

  if (row2lastSize == 0) {
    t2->GetEvent(t2->GetEntries() - 2);
    row2lastSize = row2.size();
    while (row2[row2lastSize - 1].size() == 0 && row2lastSize > 0) {
      row2lastSize--;
      nro2--;
    }
  }

  if (nro1 != nro2) {
    printf("N readout windows different!!!! %d != %d \n", nro1, nro2);
    status = false;
    return status;
  }

  printf("N readout windows = %d\n", nro1);

  int offset = 0;
  t2->GetEvent(0);
  int nro2c = row2.size();
  int next = 1;

  for (int k = 0; k < nro1; k++) {
    if (k >= nro2c) {
      offset = nro2c;
      t2->GetEvent(next);
      nro2c += row2.size();
      next++;
    }
    int i = k;
    int i2 = i - offset;

    if (row1[i].size() != row2[i2].size()) {
      printf("Readout window %d)  different number of digits in this window!!!! %d != %d \n", i, int(row1[i].size()), int(row2[i2].size()));
      status = false;
      return status;
    }

    auto digitsRO1 = row1.at(i).getBunchChannelData(digits1);
    auto digitsRO2 = row2.at(i2).getBunchChannelData(digits2);

    for (int j = 0; j < row1[i].size(); j++) {
      bool digitstatus = true;
      if (digitsRO1[j].getChannel() != digitsRO2[j].getChannel()) {
        printf("RO %d - Digit %d/%d) Different channel number %d != %d \n", i, j, row1[i].size(), digitsRO1[j].getChannel(), digitsRO2[j].getChannel());
        digitstatus = false;
      }

      if (digitsRO1[j].getTDC() != digitsRO2[j].getTDC()) {
        printf("RO %d - Digit %d/%d) Different TDCs %d != %d \n", i, j, row1[i].size(), digitsRO1[j].getTDC(), digitsRO2[j].getTDC());
        digitstatus = false;
      }

      if (digitsRO1[j].getBC() != digitsRO2[j].getBC()) {
        printf("RO %d - Digit %d/%d) Different BCs %lu != %lu \n", i, j, row1[i].size(), digitsRO1[j].getBC(), digitsRO2[j].getBC());
        digitstatus = false;
      }

      if (digitsRO1[j].getTOT() != digitsRO2[j].getTOT()) {
        printf("RO %d - Digit %d/%d) Different TOTs %d != %d \n", i, j, row1[i].size(), digitsRO1[j].getTOT(), digitsRO2[j].getTOT());
        digitstatus = false;
      }

      if (digitsRO1[j].getTriggerBunch() != digitsRO2[j].getTriggerBunch()) {
        printf("RO %d - Digit %d/%d) Different Trigger bunches %d != %d \n", i, j, row1[i].size(), digitsRO1[j].getTriggerBunch(), digitsRO2[j].getTriggerBunch());
        digitstatus = false;
      }

      if (digitsRO1[j].getTriggerOrbit() != digitsRO2[j].getTriggerOrbit()) {
        printf("RO %d - Digit %d/%d) Different Trigger orbits %d != %d \n", i, j, row1[i].size(), digitsRO1[j].getTriggerOrbit(), digitsRO2[j].getTriggerOrbit());
        digitstatus = false;
      }

      if (digitsRO1[j].isProblematic() != digitsRO2[j].isProblematic()) {
        printf("RO %d - Digit %d/%d) Different Problematic status %d != %d \n", i, j, row1[i].size(), digitsRO1[j].isProblematic(), digitsRO2[j].isProblematic());
        digitstatus = false;
      }

      if (!digitstatus) {
        status = false;
        nfake++;
      } else
        ngood++;
    }
  }

  printf("Digits good = %d\n", ngood);
  printf("Digits fake = %d\n", nfake);

  return status;
}
