// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_LTUPARAM_H
#define O2_TRD_LTUPARAM_H

#include <iosfwd>
#include <array>

namespace o2
{
namespace trd
{

class LTUParam
{
 public:
  LTUParam();
  ~LTUParam();

  // configuration settings
  // called with special SCSN commands
  void setPtMin(int data)
  {
    mPtMin = float(data) / 1000.;
    mInvPtMin = 1 / mPtMin;
  }
  void setMagField(int data) { mMagField = float(data) / 1000.; }
  void setOmegaTau(int data) { mOmegaTau = float(data) / 1.e6; }
  void setNtimebins(int data) { mNtimebins = data; }
  void setScaleQ0(int data) { mScaleQ0 = data; }
  void setScaleQ1(int data) { mScaleQ1 = data; }
  void setLengthCorrectionEnable(int data) { mPidTracklengthCorr = bool(data); }
  void setTiltCorrectionEnable(int data) { mTiltCorr = bool(data); }
  void setPIDgainCorrectionEnable(bool data) { mPidGainCorr = data; }

  // set values directly
  void setRawPtMin(float data) { mPtMin = data; }
  void setRawMagField(float data) { mMagField = data; }
  void setRawOmegaTau(float data) { mOmegaTau = data; }
  void setRawNtimebins(int data) { mNtimebins = data; }
  void setRawScaleQ0(int data) { mScaleQ0 = data; }
  void setRawScaleQ1(int data) { mScaleQ1 = data; }
  void setRawLengthCorrectionEnable(bool data) { mPidTracklengthCorr = data; }
  void setRawTiltCorrectionEnable(bool data) { mTiltCorr = data; }
  void setRawPIDgainCorrectionEnable(bool data) { mPidGainCorr = data; }

  // retrieve the calculated information
  // which is written to the TRAPs
  int getDyCorrection(int det, int rob, int mcm) const;
  void getDyRange(int det, int rob, int mcm, int ch, int& dyMinInt, int& dyMaxInt) const;
  void getCorrectionFactors(int det, int rob, int mcm, int ch,
                            unsigned int& cor0, unsigned int& cor1, float gain = 1.) const;
  int getNtimebins() const;

  float getX(int det, int rob, int mcm) const;
  float getLocalY(int det, int rob, int mcm, int ch) const;
  float getLocalZ(int det, int rob, int mcm) const;

  float getDist(int det, int rob, int mcm, int ch) const;
  float getElongation(int det, int rob, int mcm, int) const;
  float getPhi(int det, int rob, int mcm, int ch) const;
  float getPerp(int det, int rob, int mcm, int ch) const;

 protected:
  // geometry constants
  static std::array<float, 30> mgZrow;            // z-position of pad row edge 6x5
  static std::array<float, 6> mgX;                // x-position for all layers
  static std::array<float, 6> mgInvX;             // inverse x-position for all layers (to remove divisions)
  static std::array<float, 6> mgTiltingAngle;     // tilting angle for every layer
  static std::array<float, 6> mgTiltingAngleTan;  // tan of tilting angle for every layer (look up table to avoid tan calculations)
  static std::array<float, 6> mgWidthPad;         // pad width for all layers
  static std::array<float, 6> mgInvWidthPad;      // inverse pad width for all layers (to remove divisions)
  static float mgLengthInnerPadC0;                // inner pad length C0 chamber
  static float mgLengthOuterPadC0;                // outer pad length C0 chamber
  static std::array<float, 6> mgLengthInnerPadC1; // inner pad length C1 chambers
  static std::array<float, 6> mgLengthOuterPadC1; // outer pad length C1 chambers
  static float mgScalePad;                        // scaling factor for pad width
  static float mgDriftLength;                     // length of the  parse gaintbl Krypton_2009-01 drift region
  static float mgBinDy;                           // bin in dy (140 um)
  static int mgDyMax;                             // max dy for a tracklet (hard limit)
  static int mgDyMin;                             // min dy for a tracklet (hard limit)

  // settings
  float mMagField;          // magnetic field
  float mOmegaTau;          // omega tau, i.e. tan(Lorentz angle)
  float mPtMin;             // min. pt for deflection cut
  float mInvPtMin;          // min. pt for deflection cut (Inverted to remove division)
  int mNtimebins;           // drift time in units of timebins << 5n
  unsigned int mScaleQ0;    // scale factor for accumulated charge Q0
  unsigned int mScaleQ1;    // scale factor for accumulated charge Q1
  bool mPidTracklengthCorr; // enable tracklet length correction
  bool mTiltCorr;           // enable tilt correction
  bool mPidGainCorr;        // enable MCM gain correction factor for PID
};
} //end namespace trd
} //end namespace o2
#endif
