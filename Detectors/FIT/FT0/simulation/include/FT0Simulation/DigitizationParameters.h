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

namespace o2::ft0
{

struct DigitizationParameters {
  int nCellsA = Geometry::NCellsA;
  int nCellsC = Geometry::NCellsC;
  float zDetA = Geometry::ZdetA;
  float zDetC = Geometry::ZdetC;
  float bunchWidth = 25; //ns
  float channelWidth = Geometry::ChannelWidth;
  float ChannelWidthInverse = 0.076804916;                   // channel width in ps inverse
  float mMCPs = (Geometry::NCellsA + Geometry::NCellsC) * 4; //number of MCPs
  float mCFD_trsh = 3.;                                      // [mV]
  float mAmp_trsh = 110;                                     // [mV]
  int mTime_trg_gate = 153;                                  //4000/13;   #channels
  float mTimeDiffAC = (Geometry::ZdetA - Geometry::ZdetC) * TMath::C();
  float C_side_cable_cmps = 2.86;   //ns
  float A_side_cable_cmps = 11.020; //ns
  int mSignalWidth = 378;           //5000.ps/13.2ps   #channels
  int mtrg_central_trh = 200.;      // channels
  int mtrg_semicentral_trh = 100.;  // channels
  int mtrg_vertex = 230;            //3000./13.  #channels
  float mMip_in_V = 7;              //MIP to mV
  float mPe_in_mip = 0.004;         // invserse Np.e. in MIP 1./250.
  float mCfdShift = 1.66;           //ns
  float mCFDShiftPos = 1.47;        //// shift positive part of CFD signal; distance between 0.3 of max amplitude  to max
  float mCFDdeadTime = 15.6;        // ns
  float AmpIntegrationTime = 19;    //ns
  float IntegWindowDelayA = 6;      // ns, A side
  float IntegWindowDelayC = -1.6;   // ns, C side
  float charge2amp = 0.22;
  float mNoiseVar = 0.1;            //noise level
  float mNoisePeriod = 1 / 0.9;     // GHz low frequency noise period;
  float mV_2_Nchannels = 2.2857143; //7 mV ->16channels
};
} // namespace o2::ft0
#endif
