// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FT0_DIGITIZATION_PARAMETERS
#define ALICEO2_FT0_DIGITIZATION_PARAMETERS
#include <FT0Base/Geometry.h>
#include <CommonUtils/ConfigurableParamHelper.h>

namespace o2::ft0
{

struct DigitizationParameters
  : o2::conf::ConfigurableParamHelper<DigitizationParameters> {
  float mBunchWidth = 25;                                    //ns
  float mChannelWidthInverse = 0.076804916;                  // channel width in ps inverse

  float mMCPs = Geometry::Nchannels;                         //number of MCPs
  float mCFD_trsh = 3.;                                      // [mV]
  float mAmp_trsh = 100;                                     // [ph.e]
  float mAmpRecordLow = -4;                                  // integrate charge from
  float mAmpRecordUp = 15;                                   // to [ns]
  float mC_side_cable_cmps = 2.86;   //ns
  float mA_side_cable_cmps = 11.110; //ns
  int mtrg_central_trh = 600.;       // channels
  int mtrg_semicentral_trh = 300.;   // channels

  float mMip_in_V = 7;       //MIP to mV
  float mPe_in_mip = 0.004;  // invserse Np.e. in MIP 1./250.
  float mCfdShift = 1.66;    //ns
  float mCFDShiftPos = 1.47; //// shift positive part of CFD signal; distance between 0.3 of max amplitude  to max
  float mCFDdeadTime = 15.6; // ns
  float mCharge2amp = 0.22;
  float mNoiseVar = 0.1;            //noise level
  float mNoisePeriod = 1 / 0.9;     // GHz low frequency noise period;
  short mTime_trg_gate = 192;       // #channels

  static constexpr float mMV_2_Nchannels = 2.2857143;          //amplitude channel 7 mV ->16channels
  static constexpr float mMV_2_NchannelsInverse = 0.437499997; //inverse amplitude channel 7 mV ->16channels

  O2ParamDef(DigitizationParameters, "DigitizationParameters");
};
} // namespace o2::ft0
#endif
