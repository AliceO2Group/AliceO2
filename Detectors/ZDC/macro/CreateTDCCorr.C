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
#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"
#include <string>
#include <TH1.h>
#include <TFile.h>
#include <TRandom.h>
#endif

#include "ZDCBase/Constants.h"
#include "ZDCReconstruction/ZDCTDCCorr.h"
#include "TDCCorrPar.h"

using namespace std;

void CreateTDCCorr(long tmin = 0, long tmax = -1, std::string ccdbHost = "http://alice-ccdb.cern.ch:8080")
{
  o2::zdc::ZDCTDCCorr conf;
  int ipos = 0;
  for (int32_t itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    for (int32_t ibuk = 0; ibuk < o2::zdc::NBucket; ibuk++) {
      for (int32_t ipar = 0; ipar < o2::zdc::NFParA; ipar++) {
        conf.mAmpSigCorr[itdc][ibuk][ipar] = o2::zdc::fit_as_par_sig[ipos];
        ipos++;
      }
    }
  }
  ipos = 0;
  for (int32_t itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    for (int32_t ibun = 0; ibun < o2::zdc::NBCAn; ibun++) {
      // N.B. There is an ordering by signal in the flat file
      for (int32_t ibuks = 0; ibuks < o2::zdc::NBucket; ibuks++) {
        for (int32_t ibukb = 0; ibukb < o2::zdc::NBucket; ibukb++) {
          for (int32_t ipar = 0; ipar < o2::zdc::NFParA; ipar++) {
            conf.mAmpCorr[itdc][ibun][ibukb][ibuks][ipar] = o2::zdc::fit_as_par[ipos];
            ipos++;
          }
        }
      }
    }
  }
  ipos = 0;
  for (int32_t itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    for (int32_t ibun = 0; ibun < o2::zdc::NBCAn; ibun++) {
      // N.B. There is an ordering by signal in the flat file
      for (int32_t ibuks = 0; ibuks < o2::zdc::NBucket; ibuks++) {
        for (int32_t ibukb = 0; ibukb < o2::zdc::NBucket; ibukb++) {
          for (int32_t ipar = 0; ipar < o2::zdc::NFParT; ipar++) {
            conf.mTDCCorr[itdc][ibun][ibukb][ibuks][ipar] = o2::zdc::fit_ts_par[ipos];
            ipos++;
          }
        }
      }
    }
  }
  conf.print();

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
  LOG(info) << "CCDB server: " << ccdbHost;
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, o2::zdc::CCDBPathTDCCorr, metadata, tmin, tmax);

  //  return conf;
}
