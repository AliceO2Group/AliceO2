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

#include "Framework/Logger.h"
#include "ZDCReconstruction/ZDCTDCCorr.h"

using namespace o2::zdc;

void ZDCTDCCorr::print() const
{
  LOG(info) << "o2::zdc::ZDCTDCCorr: NTDCChannels = " << NTDCChannels << " NBCAn = " << NBCAn << " NBucket = " << NBucket << " NFParA = " << NFParA << " NFParT = " << NFParT;
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ibun = 0; ibun < NBCAn; ibun++) {
      int nnan = 0;
      for (int32_t ibukb = 0; ibukb < NBucket; ibukb++) {
        for (int32_t ibuks = 0; ibuks < NBucket; ibuks++) {
          for (int32_t ipar = 0; ipar < NFParA; ipar++) {
            if (isnan(mAmpCorr[itdc][ibun][ibukb][ibuks][ipar])) {
              nnan++;
            }
          }
        }
      }
      if (nnan > 0) {
        LOG(warning) << "o2::zdc::ZDCTDCCorr mAmpCorr: itdc = " << itdc << " bunch " << ibun << " unassigned = " << nnan;
      }
    }
  }
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ibun = 0; ibun < NBCAn; ibun++) {
      int nnan = 0;
      for (int32_t ibukb = 0; ibukb < NBucket; ibukb++) {
        for (int32_t ibuks = 0; ibuks < NBucket; ibuks++) {
          for (int32_t ipar = 0; ipar < NFParT; ipar++) {
            if (isnan(mTDCCorr[itdc][ibun][ibukb][ibuks][ipar])) {
              nnan++;
            }
          }
        }
      }
      if (nnan > 0) {
        LOG(warning) << "o2::zdc::ZDCTDCCorr mTDCCorr: itdc = " << itdc << " bunch " << ibun << " unassigned = " << nnan;
      }
    }
  }
}

void ZDCTDCCorr::dump() const
{
  printf("std::array<double,NTDCChannels*NBCAn*NBucket*NBucket*NFParT+1> fit_ts_par={\n");
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ibcan = 0; ibcan < NBCAn; ibcan++) {
      // N.B. There is an ordering by signal in the flat file
      for (int32_t ibuks = 0; ibuks < NBucket; ibuks++) {
        for (int32_t ibukb = 0; ibukb < NBucket; ibukb++) {
          for (int32_t ipar = 0; ipar < NFParA; ipar++) {
            if (isnan(mTDCCorr[itdc][ibcan][ibukb][ibuks][ipar])) {
              printf("std::numeric_limits<double>::quiet_NaN(),");
            } else {
              printf("%+e,", mTDCCorr[itdc][ibcan][ibukb][ibuks][ipar]);
            }
          }
          printf(" // ts%d_bc%+d_bk%d_sn%d\n", itdc, -NBCAn + ibcan, ibukb, ibuks);
        }
      }
    }
  }
  printf("std::numeric_limits<double>::quiet_NaN() // End_of_array\n");
  printf("};\n\n");

  printf("std::array<double,NTDCChannels*NBCAn*NBucket*NBucket*NFParA+1> fit_as_par={\n");
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ibcan = 0; ibcan < NBCAn; ibcan++) {
      // N.B. There is an ordering by signal in the flat file
      for (int32_t ibuks = 0; ibuks < NBucket; ibuks++) {
        for (int32_t ibukb = 0; ibukb < NBucket; ibukb++) {
          for (int32_t ipar = 0; ipar < NFParA; ipar++) {
            if (isnan(mAmpCorr[itdc][ibcan][ibukb][ibuks][ipar])) {
              printf("std::numeric_limits<double>::quiet_NaN(),");
            } else {
              printf("%+e,", mAmpCorr[itdc][ibcan][ibukb][ibuks][ipar]);
            }
          }
          printf(" // as%d_bc%+d_bk%d_sn%d\n", itdc, -NBCAn + ibcan, ibukb, ibuks);
        }
      }
    }
  }
  printf("std::numeric_limits<double>::quiet_NaN() // End_of_array\n");
  printf("};\n\n");

  // Corrections for single signal
  printf("std::array<double,o2::zdc::ZDCTDCCorr::NParExtC*o2::zdc::NTDCChannels+1> ts_beg_c={\n");
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ipar = 0; ipar < NParExtC; ipar++) {
      if (isnan(mTSBegC[itdc][ipar])) {
        printf("std::numeric_limits<double>::quiet_NaN(),");
      } else {
        printf("%+e,", mTSBegC[itdc][ipar]);
      }
    }
    printf(" // ts_beg_c_%d\n", itdc);
  }
  printf("std::numeric_limits<double>::quiet_NaN() // End_of_array\n");
  printf("};\n\n");

  printf("std::array<double,o2::zdc::ZDCTDCCorr::NParMidC*o2::zdc::NTDCChannels+1> ts_mid_c={\n");
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ipar = 0; ipar < NParMidC; ipar++) {
      if (isnan(mTSMidC[itdc][ipar])) {
        printf("std::numeric_limits<double>::quiet_NaN(),");
      } else {
        printf("%+e,", mTSMidC[itdc][ipar]);
      }
    }
    printf(" // ts_mid_c_%d\n", itdc);
  }
  printf("std::numeric_limits<double>::quiet_NaN() // End_of_array\n");
  printf("};\n\n");

  printf("std::array<double,o2::zdc::ZDCTDCCorr::NParExtC*o2::zdc::NTDCChannels+1> ts_end_c={\n");
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ipar = 0; ipar < NParExtC; ipar++) {
      if (isnan(mTSEndC[itdc][ipar])) {
        printf("std::numeric_limits<double>::quiet_NaN(),");
      } else {
        printf("%+e,", mTSEndC[itdc][ipar]);
      }
    }
    printf(" // ts_end_c_%d\n", itdc);
  }
  printf("std::numeric_limits<double>::quiet_NaN() // End_of_array\n");
  printf("};\n\n");

  printf("std::array<double,o2::zdc::ZDCTDCCorr::NParExtC*o2::zdc::NTDCChannels+1> af_beg_c={\n");
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ipar = 0; ipar < NParExtC; ipar++) {
      if (isnan(mAFBegC[itdc][ipar])) {
        printf("std::numeric_limits<double>::quiet_NaN(),");
      } else {
        printf("%+e,", mAFBegC[itdc][ipar]);
      }
    }
    printf(" // af_beg_c_%d\n", itdc);
  }
  printf("std::numeric_limits<double>::quiet_NaN() // End_of_array\n");
  printf("};\n\n");

  printf("std::array<double,o2::zdc::ZDCTDCCorr::NParMidC*o2::zdc::NTDCChannels+1> af_mid_c={\n");
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ipar = 0; ipar < NParMidC; ipar++) {
      if (isnan(mAFMidC[itdc][ipar])) {
        printf("std::numeric_limits<double>::quiet_NaN(),");
      } else {
        printf("%+e,", mAFMidC[itdc][ipar]);
      }
    }
    printf(" // af_mid_c_%d\n", itdc);
  }
  printf("std::numeric_limits<double>::quiet_NaN() // End_of_array\n");
  printf("};\n\n");

  printf("std::array<double,o2::zdc::ZDCTDCCorr::NParExtC*o2::zdc::NTDCChannels+1> af_end_c={\n");
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ipar = 0; ipar < NParExtC; ipar++) {
      if (isnan(mAFEndC[itdc][ipar])) {
        printf("std::numeric_limits<double>::quiet_NaN(),");
      } else {
        printf("%+e,", mAFEndC[itdc][ipar]);
      }
    }
    printf(" // af_end_c_%d\n", itdc);
  }
  printf("std::numeric_limits<double>::quiet_NaN() // End_of_array\n");
  printf("};\n\n");

}

void ZDCTDCCorr::clear()
{
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    for (int32_t ibun = 0; ibun < NBCAn; ibun++) {
      for (int32_t ibukb = 0; ibukb < NBucket; ibukb++) {
        for (int32_t ibuks = 0; ibuks < NBucket; ibuks++) {
          for (int32_t ipar = 0; ipar < NFParA; ipar++) {
            mAmpCorr[itdc][ibun][ibukb][ibuks][ipar] = std::numeric_limits<double>::quiet_NaN();
          }
          for (int32_t ipar = 0; ipar < NFParT; ipar++) {
            mTDCCorr[itdc][ibun][ibukb][ibuks][ipar] = std::numeric_limits<double>::quiet_NaN();
          }
        }
      }
    }
  }
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    // TDC time correction, constant, beginning of sequence
    for (int32_t ipar = 0; ipar < NParExtC; ipar++) {
      mTSBegC[itdc][ipar] = std::numeric_limits<double>::quiet_NaN();
    }
    // TDC time correction, constant, mid of sequence
    for (int32_t ipar = 0; ipar < NParMidC; ipar++) {
      mTSMidC[itdc][ipar] = std::numeric_limits<double>::quiet_NaN();
    }
    // TDC time correction, constant, end of sequence
    for (int32_t ipar = 0; ipar < NParExtC; ipar++) {
      mTSEndC[itdc][ipar] = std::numeric_limits<double>::quiet_NaN();
    }
    // TDC amplitude correction, constant, beginning of sequence
    for (int32_t ipar = 0; ipar < NParExtC; ipar++) {
      mAFBegC[itdc][ipar] = std::numeric_limits<double>::quiet_NaN();
    }
    // TDC amplitude correction, constant, mid of sequence
    for (int32_t ipar = 0; ipar < NParMidC; ipar++) {
      mAFMidC[itdc][ipar] = std::numeric_limits<double>::quiet_NaN();
    }
    // TDC amplitude correction, constant, end of sequence
    for (int32_t ipar = 0; ipar < NParExtC; ipar++) {
      mAFEndC[itdc][ipar] = std::numeric_limits<double>::quiet_NaN();
    }
  }
}
