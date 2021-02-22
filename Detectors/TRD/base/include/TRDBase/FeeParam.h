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
  FeeParam(const FeeParam& p);
  virtual ~FeeParam();
  FeeParam& operator=(const FeeParam& p);
  virtual void Copy(FeeParam& p) const;

  static FeeParam* instance(); // Singleton
  static void terminate();

  // Translation from MCM to Pad and vice versa
  static int getPadRowFromMCM(int irob, int imcm);
  static int getPadColFromADC(int irob, int imcm, int iadc);
  static int getExtendedPadColFromADC(int irob, int imcm, int iadc);
  static int getMCMfromPad(int irow, int icol);
  static int getMCMfromSharedPad(int irow, int icol);
  static int getROBfromPad(int irow, int icol);
  static int getROBfromSharedPad(int irow, int icol);
  static int getRobSide(int irob);
  static int getColSide(int icol);

  // SCSN-related
  static unsigned int aliToExtAli(int rob, int aliid);                                                                 // Converts the MCM-ROB combination to the extended MCM ALICE ID (used to address MCMs on the SCSN Bus)
  static int extAliToAli(unsigned int dest, unsigned short linkpair, unsigned short rocType, int* list, int listSize); // translates an extended MCM ALICE ID to a list of MCMs
  static short chipmaskToMCMlist(unsigned int cmA, unsigned int cmB, unsigned short linkpair, int* mcmList, int listSize);
  static short getRobAB(unsigned short robsel, unsigned short linkpair); // Returns the chamber side (A=0, B=0) of a ROB

  // wiring
  static int getORI(int detector, int readoutboard);
  static int getORIinSM(int detector, int readoutboard);
  //  static void createORILookUpTable();
  static int getORIfromHCID(int hcid);
  static int getHCIDfromORI(int ori, int readoutboard); // TODO we need more info than just ori, for now readoutboard is there ... might change

  // tracklet simulation
  bool getTracklet() const { return mgTracklet; }
  static void setTracklet(bool trackletSim = true) { mgTracklet = trackletSim; }
  bool getRejectMultipleTracklets() const { return mgRejectMultipleTracklets; }
  static void setRejectMultipleTracklets(bool rej = true) { mgRejectMultipleTracklets = rej; }
  bool getUseMisalignCorr() const { return mgUseMisalignCorr; }
  static void setUseMisalignCorr(bool misalign = true) { mgUseMisalignCorr = misalign; }
  bool getUseTimeOffset() const { return mgUseTimeOffset; }
  static void setUseTimeOffset(bool timeOffset = true) { mgUseTimeOffset = timeOffset; }

  // Concerning raw data format
  int getRAWversion() const { return mRAWversion; }
  void setRAWversion(int rawver);

  inline short padMcmLUT(int index) { return mgLUTPadNumbering[index]; }

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
  static bool mgTerminated;    // Defines if this class has already been terminated

  CommonParam* mCP = nullptr; // TRD common parameters class

  static std::vector<short> mgLUTPadNumbering; // Lookup table mapping Pad to MCM
  static bool mgLUTPadNumberingFilled;         // Lookup table mapping Pad to MCM

  void createPad2MCMLookUpTable();

  // Tracklet  processing on/off
  static bool mgTracklet;                // tracklet processing
  static bool mgRejectMultipleTracklets; // only accept best tracklet if found more than once
  static bool mgUseMisalignCorr;         // add correction for mis-alignment in y
  static bool mgUseTimeOffset;           // add time offset in calculation of fit sums

  // For raw production
  int mRAWversion{3};                    // Raw data production version
  static const int mgkMaxRAWversion = 3; // Maximum raw version number supported

  // geometry constants
  static std::array<float, constants::NCHAMBERPERSEC> mgZrow;    // z-position of pad row edge 6x5
  static std::array<float, constants::NLAYER> mgX;               // x-position for all layers
  static std::array<float, constants::NLAYER> mgInvX;            // inverse x-position for all layers (to remove divisions)
  static std::array<float, constants::NLAYER> mgTiltingAngle;    // tilting angle for every layer
  static std::array<float, constants::NLAYER> mgTiltingAngleTan; // tan of tilting angle for every layer (look up table to avoid tan calculations)
  static std::array<float, constants::NLAYER> mgWidthPad;        // pad width for all layers
  static std::array<float, constants::NLAYER> mgInvWidthPad;     // inverse pad width for all layers (to remove divisions)
  static float mgLengthInnerPadC0;                // inner pad length C0 chamber
  static float mgLengthOuterPadC0;                // outer pad length C0 chamber
  static std::array<float, constants::NLAYER> mgLengthInnerPadC1; // inner pad length C1 chambers
  static std::array<float, constants::NLAYER> mgLengthOuterPadC1; // outer pad length C1 chambers
  static float mgScalePad;                        // scaling factor for pad width
  static float mgDriftLength;                     // length of the  parse gaintbl Krypton_2009-01 drift region
  static float mgBinDy;                           // bin in dy (140 um)
  static int mgDyMax;                             // max dy for a tracklet (hard limit)
  static int mgDyMin;                             // min dy for a tracklet (hard limit)
                                                  //std::array<int,30> mgAsideLUT;                          // A side LUT to map ORI to stack/layer/side
                                                  //std::array<int,30> mgCsideLUT;                          // C side LUT to map ORI to stack/layer/side

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

 private:
  FeeParam();
};

} //namespace trd
} //namespace o2
#endif
