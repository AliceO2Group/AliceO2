// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDFEEPARAM_H
#define O2_TRDFEEPARAM_H

//Forwards to standard header with protection for GPU compilation
#include "GPUCommonRtypes.h" // for ClassDef

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
//                                                                        //
//  many things now configured by AliTRDtrapConfig reflecting             //
//  the real memory structure of the TRAP (Jochen)                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TRootIoCtor;

class TRDCommonParam;
class TRDPadPlane;
class TRDGeometry;

//_____________________________________________________________________________
class TRDFeeParam
{

 public:
  TRDFeeParam(TRootIoCtor*);
  TRDFeeParam(const TRDFeeParam& p);
  virtual ~TRDFeeParam();
  TRDFeeParam& operator=(const TRDFeeParam& p);
  virtual void Copy(TRDFeeParam& p) const;

  static TRDFeeParam* instance(); // Singleton
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
  static Float_t getSamplingFrequency() { return (Float_t)mgkLHCfrequency / 4000000.0; }
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

 protected:
  static TRDFeeParam* mgInstance; // Singleton instance
  static bool mgTerminated;       // Defines if this class has already been terminated

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
 private:
  TRDFeeParam();

  ClassDefNV(TRDFeeParam, 1); // The TRD front end electronics parameter
};

} //namespace trd
} //namespace o2
#endif
