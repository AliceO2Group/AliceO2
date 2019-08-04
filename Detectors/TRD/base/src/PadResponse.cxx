// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDBase/PadResponse.h"

using namespace o2::trd;

void PadResponse::samplePRF()
{
  //
  // Samples the pad response function
  //

  constexpr int kPRFbin = 61; // arbitraty value - need documentation/ref.
  constexpr float prf[kNlayer][kPRFbin] = {
    {2.9037e-02, 3.3608e-02, 3.9020e-02, 4.5292e-02,
     5.2694e-02, 6.1362e-02, 7.1461e-02, 8.3362e-02,
     9.7063e-02, 1.1307e-01, 1.3140e-01, 1.5235e-01,
     1.7623e-01, 2.0290e-01, 2.3294e-01, 2.6586e-01,
     3.0177e-01, 3.4028e-01, 3.8077e-01, 4.2267e-01,
     4.6493e-01, 5.0657e-01, 5.4655e-01, 5.8397e-01,
     6.1767e-01, 6.4744e-01, 6.7212e-01, 6.9188e-01,
     7.0627e-01, 7.1499e-01, 7.1851e-01, 7.1499e-01,
     7.0627e-01, 6.9188e-01, 6.7212e-01, 6.4744e-01,
     6.1767e-01, 5.8397e-01, 5.4655e-01, 5.0657e-01,
     4.6493e-01, 4.2267e-01, 3.8077e-01, 3.4028e-01,
     3.0177e-01, 2.6586e-01, 2.3294e-01, 2.0290e-01,
     1.7623e-01, 1.5235e-01, 1.3140e-01, 1.1307e-01,
     9.7063e-02, 8.3362e-02, 7.1461e-02, 6.1362e-02,
     5.2694e-02, 4.5292e-02, 3.9020e-02, 3.3608e-02,
     2.9037e-02},
    {2.5478e-02, 2.9695e-02, 3.4655e-02, 4.0454e-02,
     4.7342e-02, 5.5487e-02, 6.5038e-02, 7.6378e-02,
     8.9696e-02, 1.0516e-01, 1.2327e-01, 1.4415e-01,
     1.6794e-01, 1.9516e-01, 2.2573e-01, 2.5959e-01,
     2.9694e-01, 3.3719e-01, 3.7978e-01, 4.2407e-01,
     4.6889e-01, 5.1322e-01, 5.5569e-01, 5.9535e-01,
     6.3141e-01, 6.6259e-01, 6.8882e-01, 7.0983e-01,
     7.2471e-01, 7.3398e-01, 7.3761e-01, 7.3398e-01,
     7.2471e-01, 7.0983e-01, 6.8882e-01, 6.6259e-01,
     6.3141e-01, 5.9535e-01, 5.5569e-01, 5.1322e-01,
     4.6889e-01, 4.2407e-01, 3.7978e-01, 3.3719e-01,
     2.9694e-01, 2.5959e-01, 2.2573e-01, 1.9516e-01,
     1.6794e-01, 1.4415e-01, 1.2327e-01, 1.0516e-01,
     8.9696e-02, 7.6378e-02, 6.5038e-02, 5.5487e-02,
     4.7342e-02, 4.0454e-02, 3.4655e-02, 2.9695e-02,
     2.5478e-02},
    {2.2363e-02, 2.6233e-02, 3.0782e-02, 3.6140e-02,
     4.2535e-02, 5.0157e-02, 5.9197e-02, 6.9900e-02,
     8.2707e-02, 9.7811e-02, 1.1548e-01, 1.3601e-01,
     1.5998e-01, 1.8739e-01, 2.1840e-01, 2.5318e-01,
     2.9182e-01, 3.3373e-01, 3.7837e-01, 4.2498e-01,
     4.7235e-01, 5.1918e-01, 5.6426e-01, 6.0621e-01,
     6.4399e-01, 6.7700e-01, 7.0472e-01, 7.2637e-01,
     7.4206e-01, 7.5179e-01, 7.5551e-01, 7.5179e-01,
     7.4206e-01, 7.2637e-01, 7.0472e-01, 6.7700e-01,
     6.4399e-01, 6.0621e-01, 5.6426e-01, 5.1918e-01,
     4.7235e-01, 4.2498e-01, 3.7837e-01, 3.3373e-01,
     2.9182e-01, 2.5318e-01, 2.1840e-01, 1.8739e-01,
     1.5998e-01, 1.3601e-01, 1.1548e-01, 9.7811e-02,
     8.2707e-02, 6.9900e-02, 5.9197e-02, 5.0157e-02,
     4.2535e-02, 3.6140e-02, 3.0782e-02, 2.6233e-02,
     2.2363e-02},
    {1.9635e-02, 2.3167e-02, 2.7343e-02, 3.2293e-02,
     3.8224e-02, 4.5335e-02, 5.3849e-02, 6.4039e-02,
     7.6210e-02, 9.0739e-02, 1.0805e-01, 1.2841e-01,
     1.5216e-01, 1.7960e-01, 2.1099e-01, 2.4671e-01,
     2.8647e-01, 3.2996e-01, 3.7660e-01, 4.2547e-01,
     4.7536e-01, 5.2473e-01, 5.7215e-01, 6.1632e-01,
     6.5616e-01, 6.9075e-01, 7.1939e-01, 7.4199e-01,
     7.5838e-01, 7.6848e-01, 7.7227e-01, 7.6848e-01,
     7.5838e-01, 7.4199e-01, 7.1939e-01, 6.9075e-01,
     6.5616e-01, 6.1632e-01, 5.7215e-01, 5.2473e-01,
     4.7536e-01, 4.2547e-01, 3.7660e-01, 3.2996e-01,
     2.8647e-01, 2.4671e-01, 2.1099e-01, 1.7960e-01,
     1.5216e-01, 1.2841e-01, 1.0805e-01, 9.0739e-02,
     7.6210e-02, 6.4039e-02, 5.3849e-02, 4.5335e-02,
     3.8224e-02, 3.2293e-02, 2.7343e-02, 2.3167e-02,
     1.9635e-02},
    {1.7224e-02, 2.0450e-02, 2.4286e-02, 2.8860e-02,
     3.4357e-02, 4.0979e-02, 4.8966e-02, 5.8612e-02,
     7.0253e-02, 8.4257e-02, 1.0102e-01, 1.2094e-01,
     1.4442e-01, 1.7196e-01, 2.0381e-01, 2.4013e-01,
     2.8093e-01, 3.2594e-01, 3.7450e-01, 4.2563e-01,
     4.7796e-01, 5.2991e-01, 5.7974e-01, 6.2599e-01,
     6.6750e-01, 7.0344e-01, 7.3329e-01, 7.5676e-01,
     7.7371e-01, 7.8410e-01, 7.8793e-01, 7.8410e-01,
     7.7371e-01, 7.5676e-01, 7.3329e-01, 7.0344e-01,
     6.6750e-01, 6.2599e-01, 5.7974e-01, 5.2991e-01,
     4.7796e-01, 4.2563e-01, 3.7450e-01, 3.2594e-01,
     2.8093e-01, 2.4013e-01, 2.0381e-01, 1.7196e-01,
     1.4442e-01, 1.2094e-01, 1.0102e-01, 8.4257e-02,
     7.0253e-02, 5.8612e-02, 4.8966e-02, 4.0979e-02,
     3.4357e-02, 2.8860e-02, 2.4286e-02, 2.0450e-02,
     1.7224e-02},
    {1.5096e-02, 1.8041e-02, 2.1566e-02, 2.5793e-02,
     3.0886e-02, 3.7044e-02, 4.4515e-02, 5.3604e-02,
     6.4668e-02, 7.8109e-02, 9.4364e-02, 1.1389e-01,
     1.3716e-01, 1.6461e-01, 1.9663e-01, 2.3350e-01,
     2.7527e-01, 3.2170e-01, 3.7214e-01, 4.2549e-01,
     4.8024e-01, 5.3460e-01, 5.8677e-01, 6.3512e-01,
     6.7838e-01, 7.1569e-01, 7.4655e-01, 7.7071e-01,
     7.8810e-01, 7.9871e-01, 8.0255e-01, 7.9871e-01,
     7.8810e-01, 7.7071e-01, 7.4655e-01, 7.1569e-01,
     6.7838e-01, 6.3512e-01, 5.8677e-01, 5.3460e-01,
     4.8024e-01, 4.2549e-01, 3.7214e-01, 3.2170e-01,
     2.7527e-01, 2.3350e-01, 1.9663e-01, 1.6461e-01,
     1.3716e-01, 1.1389e-01, 9.4364e-02, 7.8109e-02,
     6.4668e-02, 5.3604e-02, 4.4515e-02, 3.7044e-02,
     3.0886e-02, 2.5793e-02, 2.1566e-02, 1.8041e-02,
     1.5096e-02}};

  // More sampling precision with linear interpolation
  std::array<float, kPRFbin> pad{};
  int sPRFbin = kPRFbin;
  float sPRFwid = (mPRFhi - mPRFlo) / ((float)sPRFbin);
  for (int iPad = 0; iPad < sPRFbin; iPad++) {
    pad[iPad] = ((float)iPad + 0.5) * sPRFwid + mPRFlo;
  }

  mPRFwid = (mPRFhi - mPRFlo) / ((float)mPRFbin);
  mPRFpad = ((int)(1.0 / mPRFwid));

  int ipos1;
  int ipos2;
  float diff;

  for (int iLayer = 0; iLayer < kNlayer; ++iLayer) {
    for (int iBin = 0; iBin < mPRFbin; ++iBin) {
      float bin = (((float)iBin) + 0.5) * mPRFwid + mPRFlo;
      ipos1 = ipos2 = 0;
      diff = 0;
      do {
        diff = bin - pad[ipos2++];
      } while ((diff > 0) && (ipos2 < kPRFbin));
      if (ipos2 == kPRFbin) {
        mPRFsmp[iLayer * mPRFbin + iBin] = prf[iLayer][ipos2 - 1];
      } else if (ipos2 == 1) {
        mPRFsmp[iLayer * mPRFbin + iBin] = prf[iLayer][ipos2 - 1];
      } else {
        --ipos2;
        if (ipos2 >= kPRFbin)
          ipos2 = kPRFbin - 1;
        ipos1 = ipos2 - 1;
        mPRFsmp[iLayer * mPRFbin + iBin] = prf[iLayer][ipos2] + diff * (prf[iLayer][ipos2] - prf[iLayer][ipos1]) / sPRFwid;
      }
    }
  }
}

int PadResponse::getPRF(double signal, double dist, int layer, double* pad) const
{
  //
  // Applies the pad response
  //
  int iBin = ((int)((-dist - mPRFlo) / mPRFwid));
  int iOff = layer * mPRFbin;

  int iBin0 = iBin - mPRFpad + iOff;
  int iBin1 = iBin + iOff;
  int iBin2 = iBin + mPRFpad + iOff;

  pad[0] = 0;
  pad[1] = 0;
  pad[2] = 0;

  if ((iBin1 >= 0) && (iBin1 < (mPRFbin * kNlayer))) {
    if (iBin0 >= 0) {
      pad[0] = signal * mPRFsmp[iBin0];
    }
    pad[1] = signal * mPRFsmp[iBin1];
    if (iBin2 < (mPRFbin * kNlayer)) {
      pad[2] = signal * mPRFsmp[iBin2];
    }
    return 1;
  } else {
    return 0;
  }
}