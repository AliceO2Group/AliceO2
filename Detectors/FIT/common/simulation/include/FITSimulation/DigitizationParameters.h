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

#ifndef ALICEO2_FIT_DIGITIZATION_PARAMETERS
#define ALICEO2_FIT_DIGITIZATION_PARAMETERS

namespace o2::fit
{
struct DigitizationParameters {
  int NCellsA;        // number of radiatiors on A side
  int NCellsC;        // number of radiatiors on C side
  float ZdetA;        // Z position A side
  float ZdetC;        // Z position on C side
  float ChannelWidth; // channel width in ps

  Float_t mBC_clk_center; // clk center
  Int_t mMCPs;            //number of MCPs
  Float_t mCFD_trsh_mip;  // = 4[mV] / 10[mV/mip]
  Float_t mTime_trg_gate; // ns
  bool mIsFT0;          //amplitude FT0(true) or FV0 (false)
  Float_t mSignalWidth; // CFD gate in ns
  Float_t mCfdShift;    // time shift for CFD shape simulation
  Float_t mMip_in_V;    //MIP to mV
  Float_t mPe_in_mip;   // Np.e. in MIP

  double mCFDShiftPos; // shift positive part of CFD signal; distance between 0.3 of max amplitude  to max
  double mNoiseVar;    //noise level
  double mNoisePeriod; // low frequency noise period;
};
} // namespace o2::fit
#endif
