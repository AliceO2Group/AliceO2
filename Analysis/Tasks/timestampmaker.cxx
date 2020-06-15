// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// #include "DetectorsRaw/HBFUtils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBQuery.h"
#include <map>
#include "TFile.h"
#include "TKey.h"
#include <iostream>

#include <CCDB/CcdbApi.h>

#include <chrono>
#include <thread>

class TSvsRun : public TNamed
{
 public:
  TSvsRun() = default;
  ~TSvsRun() = default;

  template <typename It>
  void printInsertionStatus(It it, bool success)
  {
    std::cout << "Insertion of " << it->first << (success ? " succeeded\n" : " failed\n");
  }

  Bool_t Has(Int_t run) { return mapping.count(run); }

  void Insert(Int_t run, long timestamp)
  {
    if (Has(run))
      return;
    const auto [it_hinata, success] = mapping.insert({run, timestamp});
    printInsertionStatus(it_hinata, success);
  }
  long GetTimestamp(Int_t run) { return mapping.at(run); }

 private:
  std::map<int, long> mapping;
};

int main(int argc, char* argv[])
{
  Int_t run_number = 1;
  Int_t globalbc = 1;
  Int_t orbit_freq = 11;
  Int_t bcperorbit = 2;
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " CCDBFile.root \n";
  }
  TFile file(argv[1]);

  // TFilre f("/tmp/tsmaker.root", "RECREATE");
  TSvsRun t;
  long timestamp = 1; // 11khz * bc.globalBC() * #bcperorbit;
  t.Insert(run_number, timestamp);

  return 0;
}
