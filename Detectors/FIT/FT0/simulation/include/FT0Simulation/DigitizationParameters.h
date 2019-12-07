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
  float channelWidth = Geometry::ChannelWidth;
  float mMCPs = (Geometry::NCellsA + Geometry::NCellsC) * 4; //number of MCPs
  float mCFD_trsh = 3.;                                      // [mV]
  int mTime_trg_gate = 307;                                  //4000/13;   #channels
  float mTimeDiffAC = (Geometry::ZdetA - Geometry::ZdetC) * TMath::C();
  float C_side_cable_cmps = 2.86;   //ns
  float A_side_cable_cmps = 11.020; //ns
  int mSignalWidth = 5;
  int mtrg_central_trh = 200.;     // channels
  int mtrg_semicentral_trh = 100.; // channels
  int mtrg_vertex = 230;           //3000./13.  #channels
  float mCfdShift = 1.66;       //ns
  float mMip_in_V = 7;          //MIP to mV
  float mPe_in_mip = 250;       // Np.e. in MIP
  float mCFDShiftPos = 1.47;    //// shift positive part of CFD signal; distance between 0.3 of max amplitude  to max
  float mNoiseVar = 0.1;        //noise level
  float mNoisePeriod = 1 / 0.9; // GHz low frequency noise period;
  float mV_2_Nchannels = 2.2857143; //7 mV ->16channels
};
} // namespace o2::ft0
#endif
