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
#include <FITSimulation/DigitizationParameters.h>
#include <FT0Base/Geometry.h>

namespace o2::ft0
{
inline o2::fit::DigitizationParameters FT0DigitizationParameters()
{
  o2::fit::DigitizationParameters result;
  result.NCellsA = Geometry::NCellsA;
  result.NCellsC = Geometry::NCellsC;
  result.ZdetA = Geometry::ZdetA;
  result.ZdetC = Geometry::ZdetC;
  result.ChannelWidth = Geometry::ChannelWidth;
  result.mBC_clk_center = 12.5;                               // clk center
  result.mMCPs = (Geometry::NCellsA + Geometry::NCellsC) * 4; //number of MCPs
  result.mCFD_trsh_mip = 3.;                                  // [mV]
  result.mTime_trg_gate = 4.;                                 // ns
  result.mTimeDiffAC = (Geometry::ZdetA - Geometry::ZdetC) * TMath::C();
  result.mIsFT0 = true;
  result.mSignalWidth = 5;
  result.mCfdShift = 1.66;       //ns
  result.mMip_in_V = 7;          //MIP to mV
  result.mPe_in_mip = 250;       // Np.e. in MIP
  result.mCFDShiftPos = 1.47;    //// shift positive part of CFD signal; distance between 0.3 of max amplitude  to max
  result.mNoiseVar = 0.1;        //noise level
  result.mNoisePeriod = 1 / 0.9; // GHz low frequency noise period;

  return result;
}
} // namespace o2::ft0
#endif
