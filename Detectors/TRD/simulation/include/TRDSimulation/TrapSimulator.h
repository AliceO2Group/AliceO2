// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_TRAPSIMULATOR_H
#define O2_TRD_TRAPSIMULATOR_H

///////////////////////////////////////////////////////
//                                                   //
//  TRAP Chip Simulation Class               //
//                                                   //
///////////////////////////////////////////////////////

#include <iosfwd>
#include <iostream>
#include <ostream>
#include <fstream>
#include <gsl/span>

#include "TRDBase/FeeParam.h"
#include "TRDSimulation/Digitizer.h"
#include "TRDSimulation/TrapConfig.h"

#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Constants.h"

class TH2F;

namespace o2
{
namespace trd
{

class TrapSimulator
{
 public:
  TrapSimulator() = default;
  ~TrapSimulator() = default;

  enum { PRINTRAW = 1,
         PRINTFILTERED = 2,
         PRINTDETECTED = 4,
         PRINTFOUND = 8 };
  enum { PLOTRAW = 1,
         PLOTHITS = 2,
         PLOTTRACKLETS = 4 };

  // Initialize MCM by the position parameters
  void init(TrapConfig* trapconfig, int det, int rob, int mcm);

  bool checkInitialized() const { return mInitialized; }

  // clears filter registers and internal data
  void reset();
  //  void setDebugStream(TTreeSRedirector* stream) { mDebugStream = stream; }
  //  TTreeSRedirector* getDebugStream() const { return mDebugStream; }

  bool loadMCM(/*AliRunLoader* const runloader, */ int det, int rob, int mcm);

  void noiseTest(int nsamples, int mean, int sigma, int inputGain = 1, int inputTail = 2);

  int getDataRaw(int iadc, int timebin) const { return (mADCR[iadc * mNTimeBin + timebin]); } // >> 2); }
  // get unfiltered ADC data
  int getDataFiltered(int iadc, int timebin) const { return (mADCF[iadc * mNTimeBin + timebin]); } // >> 2); }
  // get filtered ADC data
  int getZeroSupressionMap(int iadc) const { return (mZSMap[iadc]); }
  bool isDataSet() { return mDataIsSet; };
  // set ADC data with array
  void setData(int iadc, const ArrayADC& adc, unsigned int digitIdx);
  // set ADC data with array
  //void setData(int iadc, const ArrayADC& adc, gsl::span<o2::MCCompLabel,-1>& labels);
  void setBaselines();                                                                                                              // set the baselines as done in setDataByPad which is bypassed due to using setData in line above.
  void setData(int iadc, int it, int adc);                                                                                          // set ADC data
  void setDataFromDigitizerAndRun(std::vector<o2::trd::Digit>& data, o2::dataformats::MCTruthContainer<MCLabel>&);                  // data coming in manually from the the digitizer.
  void setDataByPad(std::vector<o2::trd::Digit>& padrowdata, o2::dataformats::MCTruthContainer<MCLabel>& labels, int padrowoffset); // data coming in manually from the the digitizer.
  void setDataPedestal(int iadc);                                                                                                   // Fill ADC data with pedestal values

  static bool getApplyCut() { return mgApplyCut; }
  static void setApplyCut(bool applyCut) { mgApplyCut = applyCut; }

  static int getAddBaseline() { return mgAddBaseline; }
  static void setAddBaseline(int baseline) { mgAddBaseline = baseline; }
  // Additional baseline which is added for the processing
  // in the TRAP and removed when writing back the data.
  // This is needed to run with TRAP parameters set for a
  // different baseline but it will not change the baseline
  // of the output.

  static void setStoreClusters(bool storeClusters) { mgStoreClusters = storeClusters; }
  static bool getStoreClusters() { return mgStoreClusters; }

  int getDetector() const { return mDetector; }; // Returns Chamber ID (0-539)
  int getRobPos() const { return mRobPos; };     // Returns ROB position (0-7)
  int getMcmPos() const { return mMcmPos; };     // Returns MCM position (0-17) (16,17 are mergers)
  int getRow() const { return mRow; };           // Returns Row number on chamber where the MCM is sitting
  int getCol(int iadc);                          // get corresponding column (0-143) from for ADC channel iadc = [0:20]
  // for the ADC/Col mapping, see: http://wiki.kip.uni-heidelberg.de/ti/TRD/index.php/Image:ROB_MCM_numbering.pdf
  const int getNumberOfTimeBins() const { return mNTimeBin; };
  bool storeTracklets(); // Stores tracklets to file -- debug purposes

  int packData(std::vector<uint32_t>& rawdata, uint32_t offset);
  int getRawStream(std::vector<uint32_t>& buf, uint32_t offset, unsigned int iEv = 0) const; // Produce raw data stream - Real data format
  int getTrackletStream(std::vector<uint32_t>& buf, uint32_t offset);                        // produce the tracklet stream for this MCM

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

  int getNHits() const { return mHits.size(); }
  bool getHit(int index, int& channel, int& timebin, int& qtot, int& ypos, float& y) const;

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

  // PID
  int getPID(int q0, int q1);
  void printPidLutHuman();

  // I/O
  void printFitRegXml(std::ostream& os) const;
  void printTrackletsXml(std::ostream& os) const;
  void printAdcDatTxt(std::ostream& os) const;
  void printAdcDatHuman(std::ostream& os) const;
  void printAdcDatXml(std::ostream& os) const;
  void printAdcDatDatx(std::ostream& os, bool broadcast = kFALSE, int timeBinOffset = -1) const;

  static bool readPackedConfig(TrapConfig* cfg, int hc, unsigned int* data, int size);

  // DMEM addresses
  static const int mgkDmemAddrLUTcor0 = 0xC02A;
  static const int mgkDmemAddrLUTcor1 = 0xC028;
  static const int mgkDmemAddrLUTnbins = 0xC029;

  static const int mgkDmemAddrLUTStart = 0xC100;  // LUT start address
  static const int mgkDmemAddrLUTEnd = 0xC3FF;    // maximum possible end address for the LUT table
  static const int mgkDmemAddrLUTLength = 0xC02B; // address where real size of the LUT table is stored

  static const int mgkDmemAddrTrackletStart = 0xC0E0; // Storage area for tracklets, start address
  static const int mgkDmemAddrTrackletEnd = 0xC0E3;   // Storage area for tracklets, end address

  static const int mgkDmemAddrDeflCorr = 0xc022;     // DMEM address of deflection correction
  static const int mgkDmemAddrNdrift = 0xc025;       // DMEM address of Ndrift
  static const int mgkDmemAddrDeflCutStart = 0xc030; // DMEM start address of deflection cut
  static const int mgkDmemAddrDeflCutEnd = 0xc055;   // DMEM end address of deflection cut
  static const int mgkDmemAddrTimeOffset = 0xc3fe;   // DMEM address of time offset t0
  static const int mgkDmemAddrYcorr = 0xc3ff;        // DMEM address of y correction (mis-alignment)
  static const int mgkMaxTracklets = 4;              // maximum number of tracklet-words submitted per MCM (one per CPU)
  static constexpr int mQ2Startbin = 3;              // Start range of Q2, for now here. TODO pull from a revised TrapConfig?
  static constexpr int mQ2Endbin = 5;                // End range of Q2, also pull from a revised trapconfig at some point.

  static const int mgkFormatIndex;   // index for format settings in stream
                                     //TODO should this change to 3 for the new format ????? I cant remember now, ask someone.
  static const int mgkAddDigits = 2; // additional digits used for internal representation of ADC data
                                     // all internal data as after data control block (i.e. 12 bit), s. TRAP manual
  static const int mgkNCPU = 4;      // Number of CPUs in the TRAP
  static const int mgkNHitsMC = 150; // maximum number of hits for which MC information is kept

  static const std::array<unsigned short, 4> mgkFPshifts; // shifts for pedestal filter
  // hit detection
  // individual hits can be stored as MC info
  class Hit
  { // Array of detected hits (only available in MC)
   public:
    Hit() = default;
    Hit(int channel, int timebin, int qtot, int ypos) : mChannel(channel), mTimebin(timebin), mQtot(qtot), mYpos(ypos) {}
    ~Hit() = default;
    int mChannel; // ADC channel of the hit
    int mTimebin; // timebin of the hit
    int mQtot;    // total charge of the hit
    int mYpos;    // calculated y-position
    void ClearHits()
    {
      mChannel = 0;
      mTimebin = 0;
      mQtot = 0;
      mYpos = 0;
    }
  };

  std::array<Hit, mgkNHitsMC> mHits{}; // was 100 in the run2 via fgkNHitsMC;

  class FilterReg
  {
   public:
    FilterReg() = default;
    ~FilterReg() = default;
    unsigned int mPedAcc;          // Accumulator for pedestal filter
    unsigned int mGainCounterA;    // Counter for values above FGTA in the gain filter
    unsigned int mGainCounterB;    // Counter for values above FGTB in the gain filter
    unsigned short mTailAmplLong;  // Amplitude of the long component in the tail filter
    unsigned short mTailAmplShort; // Amplitude of the short component in the tail filter
    void ClearReg()
    {
      mPedAcc = 0;
      mGainCounterA = 0;
      mGainCounterB = 0;
      mTailAmplLong = 0;
      mTailAmplShort = 0;
    };
  };
  // tracklet calculation
  class FitReg
  { // pointer to the 18 fit registers
   public:
    FitReg() = default;
    ~FitReg() = default;
    int mNhits;          // number of hits
    unsigned int mQ0;    // charge accumulated in first window
    unsigned int mQ1;    // charge accumulated in second window
    unsigned int mQ2;    // charge accumulated in third windows currently timebin 3 to 5
    unsigned int mSumX;  // sum x
    int mSumY;           // sum y
    unsigned int mSumX2; // sum x**2
    unsigned int mSumY2; // sum y**2
    int mSumXY;          // sum x*y
    void ClearReg()
    {
      mNhits = 0;
      mQ0 = 0;
      mQ1 = 0;
      mQ2 = 0; //TODO should this go here as its calculated differeintly in softwaren not hardware like the other 2?
      mSumX = 0;
      mSumY = 0;
      mSumX2 = 0;
      mSumY2 = 0;
      mSumXY = 0;
    }
    void Print()
    {
      LOG(info) << "FitReg : ";
      LOG(info) << "\t Q 0:1:2 : " << mQ0 << ":" << mQ1 << ":" << mQ2;
      LOG(info) << "\t SumX:SumY:SumX2:SumY2:SumXY : " << mSumX << ":" << mSumY << ":" << mSumX2 << ":" << mSumY2 << ":" << mSumXY;
    }
  };
  std::array<FitReg, constants::NADCMCM> mFitReg{};
  //class to store the tracklet details that are not stored in tracklet64.
  //used for later debugging purposes or in depth analysis of some part of tracklet creation or properties.
  class TrackletDetail
  {
   public:
    TrackletDetail() = default;
    TrackletDetail(float slope, float position, int q0, int q1, int q2, std::array<int, 3> hits, float error)
    {
      mSlope = slope;
      mPosition = position;
      mError = error;
      mCharges[0] = q0;
      mCharges[1] = q1;
      mCharges[2] = q2;
      mPidHits = hits;
    };
    ~TrackletDetail() = default;
    std::array<int, 3> mPidHits;    // no. of contributing clusters in each pid window.
    std::array<int, 3> mCharges;    // Q0,Q1,Q2, charges in each pid window.
    float mSlope;                   // tracklet slope
    float mPosition;                // tracklet offset
    float mError;                   // tracklet error
    int mNClusters;                 // no. of clusters
    std::vector<float> mResiduals;  //[mNClusters] cluster to tracklet residuals
    std::vector<float> mClsCharges; //[mNClusters] cluster charge
    void clear()
    {
      mPidHits[0] = 0;
      mPidHits[1] = 0;
      mPidHits[2] = 0;
      mCharges[0] = 0;
      mCharges[1] = 0;
      mCharges[2] = 0;
      mSlope = 0;
      mPosition = 0;
      mError = 0;
      mNClusters = 0;
      mResiduals.clear();
      mClsCharges.clear();
    }
    void setHits(std::array<int, 3> hits) { mPidHits = hits; }

    void setClusters(std::vector<float> res, std::vector<float> charges, int nclusters)
    {
      mResiduals = res;
      mClsCharges = charges;
      mNClusters = nclusters;
    }
  };

 protected:
  void setNTimebins(int ntimebins); // allocate data arrays corr. to the no. of timebins

  bool mInitialized{false}; // memory is allocated if initialized
  int mDetector{-1};        // Chamber ID
  int mRobPos{-1};          // ROB Position on chamber
  int mMcmPos{-1};          // MCM Position on chamber
  int mRow{-1};             // Pad row number (0-11 or 0-15) of the MCM on chamber
  int mNTimeBin{-1};        // Number of timebins currently allocated

  //TODO adcr adcf labels zerosupressionmap can all go into their own class. Refactor when stable.
  std::vector<int> mADCR; // Array with MCM ADC values (Raw, 12 bit) 2d with dimension mNTimeBin
  std::vector<int> mADCF; // Array with MCM ADC values (Filtered, 12 bit) 2d with dimension mNTimeBin
  std::array<unsigned int, constants::NADCMCM> mADCDigitIndices{}; // indices of the incoming digits, used to relate the tracklets to labels in TRDTrapSimulatorSpec
  std::vector<unsigned int> mMCMT;      // tracklet word for one mcm/trap-chip
  std::vector<Tracklet64> mTrackletArray64; // Array of 64 bit tracklets
  std::vector<unsigned short> mTrackletDigitCount; // Keep track of the number of digits contributing to the tracklet (for MC labels)
  std::vector<unsigned int> mTrackletDigitIndices; // For each tracklet the up to two global indices of the digits which contributed (global digit indices are managed in the TRDDPLTrapSimulatorTask class)
  std::vector<TrackletDetail> mTrackletDetails; // store additional tracklet information for eventual debug output.
  std::vector<int> mZSMap;              // Zero suppression map (1 dimensional projection)

  std::array<int, mgkNCPU> mFitPtr{}; // pointer to the tracklet to be calculated by CPU i

  // Parameter classes
  FeeParam* mFeeParam{FeeParam::instance()}; // FEE parameters, a singleton
  TrapConfig* mTrapConfig{nullptr};          // TRAP config
  //  CalOnlineGainTables mGainTable;

  static const int NOfAdcPerMcm = constants::NADCMCM;

  std::array<FilterReg, constants::NADCMCM> mInternalFilterRegisters;
  int mADCFilled = 0; // stores bitpattern of fillted adc, for know when to fill with pure baseline, for use with setData(int iadc, const ArrayADC& adc);
  int mNHits{0};      // Number of detected hits

  // Sort functions as in TRAP
  void sort2(unsigned short idx1i, unsigned short idx2i, unsigned short val1i, unsigned short val2i,
             unsigned short* const idx1o, unsigned short* const idx2o, unsigned short* const val1o, unsigned short* const val2o) const;
  void sort3(unsigned short idx1i, unsigned short idx2i, unsigned short idx3i,
             unsigned short val1i, unsigned short val2i, unsigned short val3i,
             unsigned short* const idx1o, unsigned short* const idx2o, unsigned short* const idx3o,
             unsigned short* const val1o, unsigned short* const val2o, unsigned short* const val3o);
  void sort6To4(unsigned short idx1i, unsigned short idx2i, unsigned short idx3i, unsigned short idx4i, unsigned short idx5i, unsigned short idx6i,
                unsigned short val1i, unsigned short val2i, unsigned short val3i, unsigned short val4i, unsigned short val5i, unsigned short val6i,
                unsigned short* const idx1o, unsigned short* const idx2o, unsigned short* const idx3o, unsigned short* const idx4o,
                unsigned short* const val1o, unsigned short* const val2o, unsigned short* const val3o, unsigned short* const val4o);
  void sort6To2Worst(unsigned short idx1i, unsigned short idx2i, unsigned short idx3i, unsigned short idx4i, unsigned short idx5i, unsigned short idx6i,
                     unsigned short val1i, unsigned short val2i, unsigned short val3i, unsigned short val4i, unsigned short val5i, unsigned short val6i,
                     unsigned short* const idx5o, unsigned short* const idx6o);

  unsigned int addUintClipping(unsigned int a, unsigned int b, unsigned int nbits) const;
  // Add a and b (unsigned) with clipping to the maximum value representable by nbits
 private:
  TrapSimulator(const TrapSimulator& m);            // not implemented
  TrapSimulator& operator=(const TrapSimulator& m); // not implemented

  static bool mgApplyCut; // apply cut on deflection length

  static int mgAddBaseline; // add baseline to the ADC values

  static bool mgStoreClusters; // whether to store all clusters in the tracklets

  bool mdebugStream = false; // whether or not to keep all the additional info for eventual dumping to a tree.

  bool mDataIsSet = false;

  std::array<TrackletMCMData, 3> mTracklets;
  TrackletMCMHeader mMCMHeader;
  static constexpr bool debugheaders = false;
};

std::ostream& operator<<(std::ostream& os, const TrapSimulator& mcm);
std::ostream& operator<<(std::ostream& os, TrapSimulator::FitReg& fitreg);

} //namespace trd
} //namespace o2
#endif
