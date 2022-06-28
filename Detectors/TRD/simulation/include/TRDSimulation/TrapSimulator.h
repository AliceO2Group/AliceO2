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

#ifndef O2_TRD_TRAPSIMULATOR_H
#define O2_TRD_TRAPSIMULATOR_H

///////////////////////////////////////////////////////
//                                                   //
//  TRAP Chip Simulation Class                       //
//                                                   //
///////////////////////////////////////////////////////

#include <iosfwd>
#include <iostream>
#include <ostream>
#include <fstream>
#include <gsl/span>

#include "TRDBase/FeeParam.h"
#include "TRDSimulation/TrapConfig.h"

#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{
namespace trd
{

class TrapSimulator
{
 public:
  enum { PRINTRAW = 1,
         PRINTFILTERED = 2,
         PRINTDETECTED = 4,
         PRINTFOUND = 8 };
  enum { PLOTRAW = 1,
         PLOTHITS = 2,
         PLOTTRACKLETS = 4 };

  // Register for the ADC filters
  struct FilterReg {
    uint32_t mPedAcc;        // Accumulator for pedestal filter
    uint32_t mGainCounterA;  // Counter for values above FGTA in the gain filter
    uint32_t mGainCounterB;  // Counter for values above FGTB in the gain filter
    uint16_t mTailAmplLong;  // Amplitude of the long component in the tail filter
    uint16_t mTailAmplShort; // Amplitude of the short component in the tail filter
    void ClearReg()
    {
      mPedAcc = 0;
      mGainCounterA = 0;
      mGainCounterB = 0;
      mTailAmplLong = 0;
      mTailAmplShort = 0;
    };
  };

  // Register for the tracklet fits
  struct FitReg {
    uint16_t nHits; // number of hits
    uint16_t q0;    // charge accumulated in first window
    uint16_t q1;    // charge accumulated in second window
    uint16_t sumX;  // sum x
    int16_t sumY;   // sum y (signed)
    uint16_t sumX2; // sum x**2
    uint16_t sumY2; // sum y**2
    int32_t sumXY;  // sum x*y (signed)
    void ClearReg()
    {
      nHits = 0;
      q0 = 0;
      q1 = 0;
      sumX = 0;
      sumY = 0;
      sumX2 = 0;
      sumY2 = 0;
      sumXY = 0;
    }
    void dumpHex(int channel) const
    {
      LOGF(info, "Channel = %i", channel);
      LOGF(info, "Nhits   = %i", nHits);
      LOGF(info, "Q1 & Q0 = 0x%08x  Q1 = %i, Q0 = %i", (q1 << 16) | q0, q1, q0);
      LOGF(info, "SumX    = 0x%08x     = %i", sumX, sumX);
      LOGF(info, "SumX2   = 0x%08x     = %i", sumX2, sumX2);
      LOGF(info, "SumY    = 0x%08x     = %i", sumY, sumY);
      LOGF(info, "SumXY   = 0x%08x     = %i", sumXY, sumXY);
      LOGF(info, "SumY2   = 0x%08x     = %i", sumY2, sumY2);
    }
    void Print() const { LOGF(info, "FitReg: nHits(%u), q0(%u), q1(%u), sumX(%u), sumX2(%u), sumY(%i), sumY2(%u), sumXY(%i)", nHits, q0, q1, sumX, sumX2, sumY, sumY2, sumXY); }
  };

  // we don't allow copies of TrapSimulator
  TrapSimulator() = default;
  TrapSimulator(const TrapSimulator&) = delete;
  TrapSimulator& operator=(const TrapSimulator&) = delete;
  ~TrapSimulator() = default;

  // Initialize MCM by the position parameters
  void init(TrapConfig* trapconfig, int det, int rob, int mcm);

  bool checkInitialized() const { return mInitialized; }

  // clears filter registers and internal data
  void reset();

  void noiseTest(int nsamples, int mean, int sigma, int inputGain = 1, int inputTail = 2);

  // get unfiltered ADC data
  int getDataRaw(int iadc, int timebin) const { return mADCR[iadc * mNTimeBin + timebin]; }
  // get filtered ADC data
  int getDataFiltered(int iadc, int timebin) const { return mADCF[iadc * mNTimeBin + timebin]; }
  int getZeroSupressionMap(int iadc) const { return (mZSMap[iadc]); }
  bool isDataSet() { return mDataIsSet; };
  // set ADC data with array
  void setData(int iadc, const ArrayADC& adc, unsigned int digitIdx);
  // set the baselines to all channels
  void setBaselines();
  // set the pedestal value to all channels
  void setDataPedestal(int iadc);
  // set an additional ADC baseline value
  void setAdditionalBaseline(int adc) { mAdditionalBaseline = adc; }
  int getAdditionalBaseline() const { return mAdditionalBaseline; }

  void setUseFloatingPointForQ() { mUseFloatingPointForQ = true; }

  int getDetector() const { return mDetector; }; // Returns Chamber ID (0-539)
  int getRobPos() const { return mRobPos; };     // Returns ROB position (0-7)
  int getMcmPos() const { return mMcmPos; };     // Returns MCM position (0-17) (16,17 are mergers)
  int getNumberOfTimeBins() const { return mNTimeBin; }; // Set via TrapConfig, but should be the same as o2::trd::constants::TIMEBINS

  // transform Tracklet64 data into raw data format (up to 4 32-bit words)
  // FIXME offset is not used, should it be removed?
  int packData(std::vector<uint32_t>& rawdata, uint32_t offset) const;

  // different stages of processing in the TRAP
  void filter();                // Apply digital filters for existing data (according to configuration)
  void zeroSupressionMapping(); // Do ZS mapping for existing data
  void tracklet();              // Run tracklet preprocessor and perform tracklet fit

  // apply individual filters to all channels and timebins
  void filterPedestal(); // Apply pedestal filter
  void filterGain();     // Apply gain filter
  void filterTail();     // Apply tail filter

  // filter initialization (resets internal registers)
  void filterPedestalInit(int baseline = 10);
  void filterGainInit();
  void filterTailInit(int baseline = -1);

  // feed single sample to individual filter
  // this changes the internal registers
  // all filters operate on (10+2)-bit values!
  unsigned short filterPedestalNextSample(int adc, int timebin, unsigned short value);
  unsigned short filterGainNextSample(int adc, unsigned short value);
  unsigned short filterTailNextSample(int adc, unsigned short value);

  // tracklet calculation
  void addHitToFitreg(int adc, unsigned short timebin, unsigned short qtot, short ypos);
  void calcFitreg();
  void trackletSelection();
  void fitTracklet();

  // getters for calculated tracklets + labels
  std::vector<Tracklet64>& getTrackletArray64() { return mTrackletArray64; }
  std::vector<unsigned short>& getTrackletDigitCount() { return mTrackletDigitCount; }
  std::vector<unsigned int>& getTrackletDigitIndices() { return mTrackletDigitIndices; }

  // data display
  void print(int choice) const;     // print stored data to stdout
  void draw(int choice, int index); // draw data (ADC data, hits and tracklets)

  friend std::ostream& operator<<(std::ostream& os, const TrapSimulator& mcm); // data output using ostream (e.g. cout << mcm;)
  static std::ostream& cfdat(std::ostream& os);                                // manipulator to activate cfdat output
  static std::ostream& raw(std::ostream& os);                                  // manipulator to activate raw output
  static std::ostream& text(std::ostream& os);                                 // manipulator to activate text output

  // I/O
  void printFitRegXml(std::ostream& os) const;
  void printTrackletsXml(std::ostream& os) const;
  void printAdcDatTxt(std::ostream& os) const;
  void printAdcDatHuman(std::ostream& os) const;
  void printAdcDatXml(std::ostream& os) const;
  void printAdcDatDatx(std::ostream& os, bool broadcast = kFALSE, int timeBinOffset = -1) const;

  static bool readPackedConfig(TrapConfig* cfg, int hc, unsigned int* data, int size);

  // DMEM addresses
  static constexpr int mgkDmemAddrLUTcor0 = 0xC02A;
  static constexpr int mgkDmemAddrLUTcor1 = 0xC028;
  static constexpr int mgkDmemAddrLUTnbins = 0xC029;

  static constexpr int mgkDmemAddrLUTStart = 0xC100;  // LUT start address
  static constexpr int mgkDmemAddrLUTEnd = 0xC3FF;    // maximum possible end address for the LUT table
  static constexpr int mgkDmemAddrLUTLength = 0xC02B; // address where real size of the LUT table is stored

  static constexpr int mgkDmemAddrTrackletStart = 0xC0E0; // Storage area for tracklets, start address
  static constexpr int mgkDmemAddrTrackletEnd = 0xC0E3;   // Storage area for tracklets, end address

  static constexpr int mgkDmemAddrDeflCorr = 0xc022;     // DMEM address of deflection correction
  static constexpr int mgkDmemAddrNdrift = 0xc025;       // DMEM address of Ndrift
  static constexpr int mgkDmemAddrDeflCutStart = 0xc030; // DMEM start address of deflection cut
  static constexpr int mgkDmemAddrDeflCutEnd = 0xc055;   // DMEM end address of deflection cut
  static constexpr int mgkDmemAddrTimeOffset = 0xc3fe;   // DMEM address of time offset t0
  static constexpr int mgkDmemAddrYcorr = 0xc3ff;        // DMEM address of y correction (mis-alignment)
  static constexpr int mQ2Startbin = 3;              // Start range of Q2, for now here. TODO pull from a revised TrapConfig?
  static constexpr int mQ2Endbin = 5;                // End range of Q2, also pull from a revised trapconfig at some point.

  static const int mgkFormatIndex;   // index for format settings in stream
                                     //TODO should this change to 3 for the new format ????? I cant remember now, ask someone.
  static const int mgkAddDigits = 2; // additional digits used for internal representation of ADC data
                                     // all internal data as after data control block (i.e. 12 bit), s. TRAP manual

  static const std::array<unsigned short, 4> mgkFPshifts; // shifts for pedestal filter

 private:
  bool mInitialized{false}; // memory is allocated if initialized
  bool mDataIsSet{false};   // flag whether input data has already been provided
  int mDetector{-1};        // Chamber ID
  int mRobPos{-1};          // ROB Position on chamber
  int mMcmPos{-1};          // MCM Position on chamber
  int mNTimeBin{-1};        // Number of timebins currently allocated
  uint32_t mMcmHeaderEmpty; // the MCM header part without the charges (this depends only on the MCM row/column)
  uint64_t mTrkltWordEmpty; // the part of the Tracklet64 which depends only on MCM row/column and detector
  bool mDontSendEmptyHeaderTrklt{false}; // flag, whether empty headers should be discarded
  int mADCFilled = 0;                    // bitpattern with ADC channels with data
  int mAdditionalBaseline = 0;           // additional baseline to be added to the ADC input values

  // PID related settings
  bool mUseFloatingPointForQ{false};   // whether to use floating point or fixed multiplier calculation of the charges
  int mScaleQ{0x10000000};             // scale the charge by 1/16
  int mSizeQ0Q1{7};                    // size of Q0 and Q1 charge windows in bit
  int mMaskQ0Q1{(1 << mSizeQ0Q1) - 1}; // bit mask 0x7f
  int mSizeQ2{6};                      // size of Q2 charge window in bit
  int mMaxQ2{(1 << mSizeQ2) - 2};      // bit mask 0x3e
  int mQ2LeftMargin{7};
  int mQ2WindowWidth{7};
  // Q = Q >> N, where N takes one of the 4 values mDynShiftx, defined below
  int mDynSize{6}; // bit for charge in the dynamical mode
  int mDynMask{(1 << mDynSize) - 1};
  int mDynShift0{2};
  int mDynShift1{4};
  int mDynShift2{6};
  int mDynShift3{8};
  int mSizeLPID{12};                                                        ///< the part of the PID which is stored with each tracklet
  int mMaskLPID{(1 << mSizeLPID) - 1};                                      ///< 0xfff
  int mSizeHPID{8};                                                         ///< the part of the PID which is stored in the MCM header
  int mEmptyHPID8{(1 << mSizeHPID) - 1};                                    ///< 0xff
  int mEmptyHPID24{mEmptyHPID8 | (mEmptyHPID8 << 8) | (mEmptyHPID8 << 16)}; ///< 0xffffff

  //TODO adcr adcf labels zerosupressionmap can all go into their own class. Refactor when stable.
  std::vector<int> mADCR; // Array with MCM ADC values (Raw, 12 bit) 2d with dimension mNTimeBin
  std::vector<int> mADCF; // Array with MCM ADC values (Filtered, 12 bit) 2d with dimension mNTimeBin
  std::array<int, constants::NADCMCM> mADCDigitIndices{}; // indices of the incoming digits, used to relate the tracklets to labels in TRDTrapSimulatorSpec
  std::array<uint32_t, 4> mMCMT;                          // tracklet words for one mcm/trap-chip (one word for each cpu)
  std::vector<Tracklet64> mTrackletArray64; // Array of 64 bit tracklets
  std::vector<unsigned short> mTrackletDigitCount; // Keep track of the number of digits contributing to the tracklet (for MC labels)
  std::vector<unsigned int> mTrackletDigitIndices; // For each tracklet the up to two global indices of the digits which contributed (global digit indices are managed in the TRDDPLTrapSimulatorTask class)
  std::vector<int> mZSMap;              // Zero suppression map (1 dimensional projection)

  std::array<int, constants::NCPU> mFitPtr{};       // pointer to the tracklet to be calculated by CPU i
  std::array<FitReg, constants::NADCMCM> mFitReg{}; // Fit register for each ADC channel
  std::array<FilterReg, constants::NADCMCM> mInternalFilterRegisters;

  // Parameter classes
  FeeParam* mFeeParam{FeeParam::instance()}; // FEE parameters, a singleton
  TrapConfig* mTrapConfig{nullptr};          // TRAP config

  // Sort functions as in TRAP
  void sort2(uint16_t idx1i, uint16_t idx2i, uint16_t val1i, uint16_t val2i,
             uint16_t* idx1o, uint16_t* idx2o, uint16_t* val1o, uint16_t* val2o) const;
  void sort3(uint16_t idx1i, uint16_t idx2i, uint16_t idx3i,
             uint16_t val1i, uint16_t val2i, uint16_t val3i,
             uint16_t* idx1o, uint16_t* idx2o, uint16_t* idx3o,
             uint16_t* val1o, uint16_t* val2o, uint16_t* val3o) const;
  void sort6To4(uint16_t idx1i, uint16_t idx2i, uint16_t idx3i, uint16_t idx4i, uint16_t idx5i, uint16_t idx6i,
                uint16_t val1i, uint16_t val2i, uint16_t val3i, uint16_t val4i, uint16_t val5i, uint16_t val6i,
                uint16_t* idx1o, uint16_t* idx2o, uint16_t* idx3o, uint16_t* idx4o,
                uint16_t* val1o, uint16_t* val2o, uint16_t* val3o, uint16_t* val4o) const;
  void sort6To2Worst(uint16_t idx1i, uint16_t idx2i, uint16_t idx3i, uint16_t idx4i, uint16_t idx5i, uint16_t idx6i,
                     uint16_t val1i, uint16_t val2i, uint16_t val3i, uint16_t val4i, uint16_t val5i, uint16_t val6i,
                     uint16_t* idx5o, uint16_t* idx6o) const;

  // Add a and b (unsigned) with clipping to the maximum value representable by nbits
  unsigned int addUintClipping(unsigned int a, unsigned int b, unsigned int nbits) const;

  // The same LUT Venelin uses for his C++ test model of the TRAP (to check agreement with his results)
  const uint16_t LUT_POS[128] = {
    0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,
    16, 16, 16, 17, 17, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 26, 26, 26, 26,
    27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 27, 27, 27, 27, 26,
    26, 26, 26, 25, 25, 25, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 19, 18, 18, 17, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 7};
};

std::ostream& operator<<(std::ostream& os, const TrapSimulator& mcm);
std::ostream& operator<<(std::ostream& os, TrapSimulator::FitReg& fitreg);

} //namespace trd
} //namespace o2
#endif
