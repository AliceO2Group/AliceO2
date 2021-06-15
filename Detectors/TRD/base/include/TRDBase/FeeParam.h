// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_FEEPARAM_H
#define O2_TRD_FEEPARAM_H


////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  TRD front end electronics parameters class                            //
//  Contains all FEE (MCM, TRAP, PASA) related                            //
//  parameters, constants, and mapping.                                   //
//                                                                        //
//  Author:                                                               //
//    Ken Oyama (oyama@physi.uni-heidelberg.de)                           //
//    Merging LTUParam in here for Run3                                   //
//    TrapChip configs remain inside the TrapConfig class reflecting      //
//  the real memory structure of the TRAP (Jochen and a tiny bit of Sean) //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "DataFormatsTRD/Constants.h"
#include "TRDBase/CommonParam.h"

#include <array>
#include <vector>

class CommonParam;
class PadPlane;
class Geometry;

namespace o2
{
namespace trd
{
//_____________________________________________________________________________
class FeeParam
{

 public:
  ~FeeParam() = default;
  FeeParam(const FeeParam&) = delete;
  FeeParam& operator=(const FeeParam&) = delete;

  static FeeParam* instance(); // Singleton

  // Translation from MCM to Pad and vice versa
  static int getPadRowFromMCM(int irob, int imcm);
  static int getPadColFromADC(int irob, int imcm, int iadc);
  static int getExtendedPadColFromADC(int irob, int imcm, int iadc);
  static int getMCMfromPad(int irow, int icol);
  static int getMCMfromSharedPad(int irow, int icol);
  static int getROBfromPad(int irow, int icol);
  static int getROBfromSharedPad(int irow, int icol);
  static int getROBSide(int irob);
  static int getColSide(int icol);

  // SCSN-related
  static unsigned int aliToExtAli(int rob, int aliid);                                                                 // Converts the MCM-ROB combination to the extended MCM ALICE ID (used to address MCMs on the SCSN Bus)
  static int extAliToAli(unsigned int dest, unsigned short linkpair, unsigned short rocType, int* list, int listSize); // translates an extended MCM ALICE ID to a list of MCMs
  static short chipmaskToMCMlist(unsigned int cmA, unsigned int cmB, unsigned short linkpair, int* mcmList, int listSize);
  static short getRobAB(unsigned short robsel, unsigned short linkpair); // Returns the chamber side (A=0, B=0) of a ROB

  // wiring
  static int getORI(int detector, int readoutboard);
  static int getORIinSM(int detector, int readoutboard);
  static void unpackORI(int link, int side, int& stack, int& layer, int& halfchamberside);
  //  void createORILookUpTable();
  static int getORIfromHCID(int hcid);
  static int getHCIDfromORI(int ori, int readoutboard); // TODO we need more info than just ori, for now readoutboard is there ... might change

  // tracklet simulation
  bool getTracklet() const { return mTracklet; }
  void setTracklet(bool trackletSim = true) { mTracklet = trackletSim; }
  bool getRejectMultipleTracklets() const { return mRejectMultipleTracklets; }
  void setRejectMultipleTracklets(bool rej = true) { mRejectMultipleTracklets = rej; }
  bool getUseMisalignCorr() const { return mUseMisalignCorr; }
  void setUseMisalignCorr(bool misalign = true) { mUseMisalignCorr = misalign; }
  bool getUseTimeOffset() const { return mUseTimeOffset; }
  void setUseTimeOffset(bool timeOffset = true) { mUseTimeOffset = timeOffset; }

  // Concerning raw data format
  int getRAWversion() const { return mRAWversion; }
  void setRAWversion(int rawver);

  short padMcmLUT(int index) { return mLUTPadNumbering[index]; }

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
  static FeeParam* mgInstance; // Singleton instance

  CommonParam* mCP{CommonParam::instance()}; // TRD common parameters class

  std::array<short, constants::NCOLUMN> mLUTPadNumbering; // Lookup table mapping Pad to MCM

  void fillPad2MCMLookUpTable();

  // Tracklet  processing on/off
  bool mTracklet{true};                 // tracklet processing
  bool mRejectMultipleTracklets{false}; // only accept best tracklet if found more than once
  bool mUseMisalignCorr{false};         // add correction for mis-alignment in y
  bool mUseTimeOffset{false};           // add time offset in calculation of fit sums

  // For raw production
  int mRAWversion{3};                    // Raw data production version
  const int mkMaxRAWversion = 3;         // Maximum raw version number supported

  // geometry constants
  std::array<float, constants::NCHAMBERPERSEC> mZrow{// z-position of pad row edge 6x5
                                                     301, 177, 53, -57, -181,
                                                     301, 177, 53, -57, -181,
                                                     315, 184, 53, -57, -188,
                                                     329, 191, 53, -57, -195,
                                                     343, 198, 53, -57, -202,
                                                     347, 200, 53, -57, -204};
  std::array<float, constants::NLAYER> mX{300.65, 313.25, 325.85, 338.45, 351.05, 363.65};  // x-position for all layers
  std::array<float, constants::NLAYER> mInvX;                                               // inverse x-position for all layers (to remove divisions)
  std::array<float, constants::NLAYER> mTiltingAngle{-2., 2., -2., 2., -2., 2.};            // tilting angle for every layer
  std::array<float, constants::NLAYER> mTiltingAngleTan;                                    // tan of tilting angle for every layer (look up table to avoid tan calculations)
  std::array<float, constants::NLAYER> mWidthPad{0.635, 0.665, 0.695, 0.725, 0.755, 0.785}; // pad width for all layers
  std::array<float, constants::NLAYER> mInvWidthPad;                                        // inverse pad width for all layers (to remove divisions)
  float mLengthInnerPadC0{9.f};                                                             // inner pad length C0 chamber
  float mLengthOuterPadC0{8.f};                                                             // outer pad length C0 chamber
  std::array<float, constants::NLAYER> mLengthInnerPadC1{7.5, 7.5, 8.0, 8.5, 9.0, 9.0};     // inner pad length C1 chambers
  std::array<float, constants::NLAYER> mLengthOuterPadC1{7.5, 7.5, 7.5, 7.5, 7.5, 8.5};     // outer pad length C1 chambers
  float mScalePad{256. * 32.};                                                              // scaling factor for pad width
  float mDriftLength{3.};                                                                   // length of the  parse gaintbl Krypton_2009-01 drift region
  // WARNING: This values for dY are valid for Run 1+2 format only
  float mBinDy{140e-4}; // bin in dy (140 um)
  int mDyMax{63};       // max dy for a tracklet (hard limit)
  int mDyMin{-64};      // min dy for a tracklet (hard limit)
                        //std::array<int,30> mAsideLUT;                          // A side LUT to map ORI to stack/layer/side
                        //std::array<int,30> mCsideLUT;                          // C side LUT to map ORI to stack/layer/side

  // settings
  float mMagField{0.f};            // magnetic field
  float mOmegaTau{0.f};            // omega tau, i.e. tan(Lorentz angle)
  float mPtMin{.1f};               // min. pt for deflection cut
  float mInvPtMin{1.f / mPtMin};   // min. pt for deflection cut (Inverted to remove division)
  int mNtimebins{20 << 5};         // drift time in units of timebins << 5n
  unsigned int mScaleQ0{0};        // scale factor for accumulated charge Q0
  unsigned int mScaleQ1{0};        // scale factor for accumulated charge Q1
  bool mPidTracklengthCorr{false}; // enable tracklet length correction
  bool mTiltCorr{false};           // enable tilt correction
  bool mPidGainCorr{false};        // enable MCM gain correction factor for PID

 private:
  FeeParam();
};

} //namespace trd
} //namespace o2
#endif
