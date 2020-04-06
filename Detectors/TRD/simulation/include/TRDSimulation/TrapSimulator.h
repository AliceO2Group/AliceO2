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

#include "TRDBase/Tracklet.h"
#include "TRDBase/FeeParam.h"
#include "TRDBase/Digit.h"
#include "TRDBase/MCLabel.h"
#include "TRDSimulation/Digitizer.h"
#include "TRDSimulation/TrapConfigHandler.h"
#include "TRDSimulation/TrapConfig.h"

class TH2F;

namespace o2
{
namespace trd
{

class TrapSimulator
{
 public:
  TrapSimulator();
  ~TrapSimulator() = default;

  enum { PRINTRAW = 1,
         PRINTFILTERED = 2,
         PRINTDETECTED = 4,
         PRINTFOUND = 8 };
  enum { PLOTRAW = 1,
         PLOTHITS = 2,
         PLOTTRACKLETS = 4 };

  void init(TrapConfig* trapconfig, int det, int rob, int mcm);
  // Initialize MCM by the position parameters

  void reset();
  // clears filter registers and internal data
  void clear();
  //  void setDebugStream(TTreeSRedirector* stream) { mDebugStream = stream; }
  //  TTreeSRedirector* getDebugStream() const { return mDebugStream; }

  bool loadMCM(/*AliRunLoader* const runloader, */ int det, int rob, int mcm);

  void noiseTest(int nsamples, int mean, int sigma, int inputGain = 1, int inputTail = 2);

  int getDataRaw(int iadc, int timebin) const { return (mADCR[iadc * mNTimeBin + timebin] >> 2); }
  // get unfiltered ADC data
  int getDataFiltered(int iadc, int timebin) const { return (mADCF[iadc * mNTimeBin + timebin] >> 2); }
  // get filtered ADC data
  int getZeroSupressionMap(int iadc) const { return (mZSMap[iadc]); }
  bool isDataSet() { return mDataIsSet; };
  void unsetData()
  {
    mDataIsSet = false;
    //if(mHits.size()>50) LOG(warn) << "mHits size is >50 ==" << mHits.size();
    //    for(int i=0;i<mNHits;i++) mHits[i].ClearHits();//I dont need to unset this as I am setting mNHits to zero
    //if(mFitReg.size()>50) LOG(warn) << "mFitReg size is >50 ==" << mFitReg.size();
    for (auto& fitreg : mFitReg)
      fitreg.ClearReg();
    mNHits = 0;
  };
  void setData(int iadc, const std::vector<int>& adc);                                                                              // set ADC data with array
  void setData(int iadc, const ArrayADC& adc);                                                                                      // set ADC data with array
  void setData(int iadc, int it, int adc);                                                                                          // set ADC data
  void setDataFromDigitizerAndRun(std::vector<o2::trd::Digit>& data, o2::dataformats::MCTruthContainer<MCLabel>&);                  // data coming in manually from the the digitizer.
                                                                                                                                    /*   void setData(TRDArrayADC* const adcArray,
               TRDdigitsManager* const digitsManager = 0x0); // set ADC data from adcArray
  void setDataByPad(const TRDArrayADC* const adcArray,
                    TRDdigitsManager* const digitsManager = 0x0); // set ADC data from adcArray
*/
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
  std::string getTrklBranchName() const { return mTrklBranchName; }
  void setTrklBranchName(std::string name) { mTrklBranchName = name; }

  int produceRawStream(unsigned int* buf, int bufsize, unsigned int iEv = 0) const; // Produce raw data stream - Real data format
  int produceTrackletStream(unsigned int* buf, int bufsize);                        // produce the tracklet stream for this MCM

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
  bool getHit(int index, int& channel, int& timebin, int& qtot, int& ypos, float& y, int& label) const;

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
  //std::array<TrackletMCM> getTrackletArray() const { return mTrackletArray; }
  std::vector<Tracklet>& getTrackletArray() { return mTrackletArray; }
  void getTracklets(std::vector<Tracklet>& TrackletStore); // place the trapsim tracklets nto the incoming vector

  bool checkInitialized() const;     // Check whether the class is initialized
  static const int mgkFormatIndex;   // index for format settings in stream
                                     //TODO should this change to 3 for the new format ????? I cant remember now, ask someone.
  static const int mgkAddDigits = 2; // additional digits used for internal representation of ADC data
                                     // all internal data as after data control block (i.e. 12 bit), s. TRAP manual
  static const int mgkNCPU = 4;      // Number of CPUs in the TRAP
  static const int mgkNHitsMC = 100; // maximum number of hits for which MC information is kept

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
                  //  std::array<int, 3> mLabel{}; // up to 3 labels (only in MC) run3 is free to have many, but does more than 1 per digit make sense.
    void ClearHits()
    {
      mChannel = 0;
      mTimebin = 0;
      mQtot = 0;
      mYpos = 0;
    }
  }; //mHits[mgkNHitsMC];
  //std::array<Hit, 50> mHits;
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
      mSumX = 0;
      mSumY = 0;
      mSumX2 = 0;
      mSumY2 = 0;
      mSumXY = 0;
    }
  };
  std::array<FitReg, 25> mFitReg{}; // TODO come back and make this 21 or 22, I cant remember now which one, so making it 25 to be safe ;-)
                                    //std::vector<FitReg,FeeParam::mgkNadcMcm> mFitReg;

 protected:
  void setNTimebins(int ntimebins); // allocate data arrays corr. to the no. of timebins

  bool mInitialized;      // memory is allocated if initialized
  int mDetector;          // Chamber ID
  int mRobPos;            // ROB Position on chamber
  int mMcmPos;            // MCM Position on chamber
  int mRow;               // Pad row number (0-11 or 0-15) of the MCM on chamber
  int mNTimeBin;          // Number of timebins currently allocated
  std::vector<int> mADCR; // Array with MCM ADC values (Raw, 12 bit) 2d with dimension mNTimeBin
  std::vector<int> mADCF; // Array with MCM ADC values (Filtered, 12 bit) 2d with dimension mNTimeBin

  std::vector<unsigned int> mMCMT;      // tracklet word for one mcm/trap-chip
  std::vector<Tracklet> mTrackletArray; // Array of TRDtrackletMCM which contains MC information in addition to the tracklet word
  std::vector<int> mZSMap;              // Zero suppression map (1 dimensional projection)

  std::array<int, mgkNCPU> mFitPtr{}; // pointer to the tracklet to be calculated by CPU i

  std::string mTrklBranchName; // name of the tracklet branch to write to

  // Parameter classes
  FeeParam* mFeeParam;     // FEE parameters
  TrapConfig* mTrapConfig; // TRAP config
  TrapConfigHandler mTrapConfigHandler;
  CalOnlineGainTables mGainTable;

  static const int NOfAdcPerMcm = 21;
  //TRDdigitsManager* mDigitsManager; // pointer to digits manager used for MC label calculation
  //  TRDArrayDictionary* mDict[3];     // pointers to label dictionaries
  // Dictionaries are now done a differentway ??? to be determined TODO
  //  std::vector<int> mDict1;
  //  std::vector<int> mDict2;
  //  std::vector<int> mDict3;
  // internal filter registers

  std::array<FilterReg, NOfAdcPerMcm> mInternalFilterRegisters;

  int mNHits; // Number of detected hits

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
  bool mDataIsSet = false;
};

std::ostream& operator<<(std::ostream& os, const TrapSimulator& mcm);
} //namespace trd
} //namespace o2
#endif
