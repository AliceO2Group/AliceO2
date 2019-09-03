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

namespace o2
{
namespace trd
{

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

#include <iosfwd>
#include <array>

class TRDCommonParam;
class TRDPadPlane;
class TRDGeometry;

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
  virtual int getPadRowFromMCM(Int_t irob, Int_t imcm) const;
  virtual int getPadColFromADC(Int_t irob, Int_t imcm, Int_t iadc) const;
  virtual int getExtendedPadColFromADC(Int_t irob, Int_t imcm, Int_t iadc) const;
  virtual int getMCMfromPad(Int_t irow, Int_t icol) const;
  virtual int getMCMfromSharedPad(Int_t irow, Int_t icol) const;
  virtual int getROBfromPad(Int_t irow, Int_t icol) const;
  virtual int getROBfromSharedPad(Int_t irow, Int_t icol) const;
  virtual int getRobSide(Int_t irob) const;
  virtual int getColSide(Int_t icol) const;

  // SCSN-related
  static unsigned int aliToExtAli(Int_t rob, Int_t aliid);                                               // Converts the MCM-ROB combination to the extended MCM ALICE ID (used to address MCMs on the SCSN Bus)
  static int extAliToAli(UInt_t dest, UShort_t linkpair, UShort_t rocType, Int_t* list, Int_t listSize); // translates an extended MCM ALICE ID to a list of MCMs
  static Short_t chipmaskToMCMlist(unsigned int cmA, UInt_t cmB, UShort_t linkpair, Int_t* mcmList, Int_t listSize);
  static Short_t getRobAB(UShort_t robsel, UShort_t linkpair); // Returns the chamber side (A=0, B=0) of a ROB

  // geometry
  static Float_t getSamplingFrequency() { return (Float_t)mgkLHCfrequency / 4000000.0; } //TODO put the 40MHz into a static variable somewhere.
  static int getNmcmRob() { return mgkNmcmRob; }
  static int getNmcmRobInRow() { return mgkNmcmRobInRow; }
  static int getNmcmRobInCol() { return mgkNmcmRobInCol; }
  static int getNrobC0() { return mgkNrobC0; }
  static int getNrobC1() { return mgkNrobC1; }
  static int getNadcMcm() { return mgkNadcMcm; }
  static int getNcol() { return mgkNcol; }
  static int getNcolMcm() { return mgkNcolMcm; }
  static int getNrowC0() { return mgkNrowC0; }
  static int getNrowC1() { return mgkNrowC1; }

  // tracklet simulation
  bool getTracklet() const { return mgTracklet; }
  static void setTracklet(bool trackletSim = kTRUE) { mgTracklet = trackletSim; }
  bool getRejectMultipleTracklets() const { return mgRejectMultipleTracklets; }
  static void setRejectMultipleTracklets(bool rej = kTRUE) { mgRejectMultipleTracklets = rej; }
  bool getUseMisalignCorr() const { return mgUseMisalignCorr; }
  static void setUseMisalignCorr(bool misalign = kTRUE) { mgUseMisalignCorr = misalign; }
  bool getUseTimeOffset() const { return mgUseTimeOffset; }
  static void setUseTimeOffset(bool timeOffset = kTRUE) { mgUseTimeOffset = timeOffset; }

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

  TRDCommonParam* mCP = nullptr; // TRD common parameters class

  static std::vector<short> mgLUTPadNumbering; // Lookup table mapping Pad to MCM
  static bool mgLUTPadNumberingFilled;         // Lookup table mapping Pad to MCM

  void createPad2MCMLookUpTable();

  // Basic Geometrical numbers
  static const int mgkLHCfrequency = 40079000; // [Hz] LHC clock
  static const int mgkNmcmRob = 16;            // Number of MCMs per ROB
  static const int mgkNmcmRobInRow = 4;        // Number of MCMs per ROB in row dir.
  static const int mgkNmcmRobInCol = 4;        // Number of MCMs per ROB in col dir.
  static const int mgkNrobC0 = 6;              // Number of ROBs per C0 chamber
  static const int mgkNrobC1 = 8;              // Number of ROBs per C1 chamber
  static const int mgkNadcMcm = 21;            // Number of ADC channels per MCM
  static const int mgkNcol = 144;              // Number of pads per padplane row
  static const int mgkNcolMcm = 18;            // Number of pads per MCM
  static const int mgkNrowC0 = 12;             // Number of Rows per C0 chamber
  static const int mgkNrowC1 = 16;             // Number of Rows per C1 chamber

  // Tracklet  processing on/off
  static bool mgTracklet;                // tracklet processing
  static bool mgRejectMultipleTracklets; // only accept best tracklet if found more than once
  static bool mgUseMisalignCorr;         // add correction for mis-alignment in y
  static bool mgUseTimeOffset;           // add time offset in calculation of fit sums

  // For raw production
  int mRAWversion{3};                    // Raw data production version
  static const int mgkMaxRAWversion = 3; // Maximum raw version number supported

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

 private:
  FeeParam();
};

} //namespace trd
} //namespace o2
#endif
