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
#include "TDCSinglePar.h"
#include "ZDCBase/Helpers.h"
using namespace std;

void CreateTDCCorr(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{
  // Shortcuts: internal, external, test, local, root

  o2::zdc::ZDCTDCCorr conf;
  int ipos = 0;
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
  // Corrections for single signal
  for (int32_t itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    // TDC time correction, constant, beginning of sequence
    for (int32_t ipar = 0; ipar < o2::zdc::ZDCTDCCorr::NParExtC; ipar++) {
      conf.mTSBegC[itdc][ipar] = ts_beg_c[o2::zdc::ZDCTDCCorr::NParExtC * itdc + ipar];
    }
    // TDC time correction, constant, mid of sequence
    for (int32_t ipar = 0; ipar < o2::zdc::ZDCTDCCorr::NParMidC; ipar++) {
      conf.mTSMidC[itdc][ipar] = ts_mid_c[o2::zdc::ZDCTDCCorr::NParMidC * itdc + ipar];
    }
    // TDC time correction, constant, end of sequence
    for (int32_t ipar = 0; ipar < o2::zdc::ZDCTDCCorr::NParExtC; ipar++) {
      conf.mTSEndC[itdc][ipar] = ts_end_c[o2::zdc::ZDCTDCCorr::NParExtC * itdc + ipar];
    }
    // TDC amplitude correction, constant, beginning of sequence
    for (int32_t ipar = 0; ipar < o2::zdc::ZDCTDCCorr::NParExtC; ipar++) {
      conf.mAFBegC[itdc][ipar] = af_beg_c[o2::zdc::ZDCTDCCorr::NParExtC * itdc + ipar];
    }
    // TDC amplitude correction, constant, mid of sequence
    for (int32_t ipar = 0; ipar < o2::zdc::ZDCTDCCorr::NParMidC; ipar++) {
      conf.mAFMidC[itdc][ipar] = af_mid_c[o2::zdc::ZDCTDCCorr::NParMidC * itdc + ipar];
    }
    // TDC amplitude correction, constant, end of sequence
    for (int32_t ipar = 0; ipar < o2::zdc::ZDCTDCCorr::NParExtC; ipar++) {
      conf.mAFEndC[itdc][ipar] = af_end_c[o2::zdc::ZDCTDCCorr::NParExtC * itdc + ipar];
    }
  }

  conf.print();
  // conf.dump();

  std::string ccdb_host = o2::zdc::ccdbShortcuts(ccdbHost, conf.Class_Name(), o2::zdc::CCDBPathTDCCorr);

  if (o2::zdc::endsWith(ccdb_host, ".root")) {
    TFile f(ccdb_host.data(), "recreate");
    f.WriteObjectAny(&conf, conf.Class_Name(), "ccdb_object");
    f.Close();
    return;
  }

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdb_host.c_str());
  LOG(info) << "CCDB server: " << api.getURL();
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, o2::zdc::CCDBPathTDCCorr, metadata, tmin, tmax);

  //  return conf;
}
