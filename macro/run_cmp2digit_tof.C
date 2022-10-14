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
#include "TOFReconstruction/Decoder.h"
#include <fairlogger/Logger.h>

#endif

// example of TOF raw data encoding from digits

void run_cmp2digit_tof(std::string inName = "cmptof.bin",             // name of the output binary file
                       std::string inpName = "tofdigitsFromRaw.root", // name of the input TOF digits
                       int verbosity = 0)                             // memory caching in Byte
{
  o2::tof::compressed::Decoder decoder;

  decoder.open(inName.c_str());
  decoder.setVerbose(verbosity);

  decoder.decode();

  std::vector<o2::tof::Digit>* alldigits = decoder.getDigitPerTimeFrame();
  printf("N digits -> %d\n", int(alldigits->size()));

  std::vector<o2::tof::ReadoutWindowData>* row = decoder.getReadoutWindowData();
  printf("N readout window -> %d\n", int(row->size()));

  int n_tof_window = row->size();
  int n_orbits = n_tof_window / 3;
  int digit_size = alldigits->size();

  //  LOG(info) << "TOF: N tof window decoded = " << n_tof_window << "(orbits = " << n_orbits << ") with " << digit_size<< " digits";
  printf("Write %s\n", inpName.c_str());
  TFile* f = new TFile(inpName.c_str(), "RECREATE");
  TTree* t = new TTree("o2sim", "o2sim");
  t->Branch("TOFDigit", &alldigits);
  t->Branch("TOFReadoutWindow", &row);
  t->Fill();
  t->Write();
  f->Close();
}
