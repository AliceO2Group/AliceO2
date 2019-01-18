#ifndef ALICEO2_FIT_DIGITIZATION_PARAMETERS
#define ALICEO2_FIT_DIGITIZATION_PARAMETERS

namespace o2::fit
{
struct DigitizationParameters {
  int NCellsA;        // number of radiatiors on A side
  int NCellsC;        // number of radiatiors on C side
  float ZdetA;        // number of radiatiors on A side
  float ZdetC;        // number of radiatiors on C side
  float ChannelWidth; // channel width in ps

  Float_t mBC_clk_center; // clk center
  Int_t mMCPs;            //number of MCPs
  Float_t mCFD_trsh_mip;  // = 4[mV] / 10[mV/mip]
  Float_t mTime_trg_gate; // ns
  Int_t mAmpThreshold;    // number of photoelectrons
  Float_t mTimeDiffAC;
};
} // namespace o2::fit
#endif
