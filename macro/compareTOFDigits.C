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

#endif

bool compareTOFDigits(std::string inpName1 = "tofdigits.root", std::string inpName2 = "tofdigitsFromRaw.root")
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

  while (row1[nro1 - 1].size() == 0 && nro1 > 0)
    nro1--;
  while (row2[nro2 - 1].size() == 0 && nro2 > 0)
    nro2--;

  if (nro1 != nro2) {
    printf("N readout windows different!!!! %d != %d \n", nro1, nro2);
    status = false;
    return status;
  }

  printf("N readout windows = %d\n", nro1);

  for (int i = 0; i < nro1; i++) {
    if (row1[i].size() != row2[i].size()) {
      printf("Readout window %d)  different number of digits in this window!!!! %d != %d \n", i, int(row1[i].size()), int(row2[i].size()));
      status = false;
      return status;
    }

    auto digitsRO1 = row1.at(i).getBunchChannelData(digits1);
    auto digitsRO2 = row2.at(i).getBunchChannelData(digits2);

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
        printf("RO %d - Digit %d/%d) Different BCs %d != %d \n", i, j, row1[i].size(), digitsRO1[j].getBC(), digitsRO2[j].getBC());
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
