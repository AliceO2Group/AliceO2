/// \file GBTFrameContainer.h
/// \brief Container class for the GBT Frames
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_GBTFRAMECONTAINER_H_
#define ALICEO2_TPC_GBTFRAMECONTAINER_H_

#include "TPCReconstruction/GBTFrame.h"
#include "TPCReconstruction/AdcClockMonitor.h"
#include "TPCReconstruction/SyncPatternMonitor.h"
#include "TPCReconstruction/HalfSAMPAData.h"
#include "TPCReconstruction/DigitData.h"
#include "TPCBase/Mapper.h" 
//#include <TClonesArray.h>  

#include <iterator>
#include <vector>
#include <queue>
#include <array>
#include <mutex>

#include <thread>

#include <fstream>
#include <iostream>
#include <iomanip>

#include "FairLogger.h"

namespace o2{
namespace TPC{

/// \class GBTFrameContainer
/// \brief GBT Frame container class

class GBTFrameContainer {
  public:

    /// Default constructor
    GBTFrameContainer();

    /// Default constructor
    /// @param cru CRU ID
    /// @param link Link ID
    GBTFrameContainer(int cru, int link);

    /// Constructor
    /// @param size Size of GBT frame container to avoid unnecessary reallocation of memory
    /// @param cru CRU ID
    /// @param link Link ID
    GBTFrameContainer(int size, int cru, int link);

    /// Destructor
    ~GBTFrameContainer();

    /// Reset function to clear container
    void reset();

    /// Get the size of the container
    /// @return Size of GBT frame container
    int getSize() { return mGBTFrames.size(); };

    /// Get the number of entries in the container
    /// @return Number of entries in the GBT frame container
    int getNentries();

    /// Add copy of frame to the container
    /// @param frame GBT Frame
    void addGBTFrame(GBTFrame& frame);

    /// Add frame to the container
    /// @param word3 Word 3 of GBT frame, contains bit [127: 96], [127:112] are not part of the actual frame
    /// @param word2 Word 2 of GBT frame, contains bit [ 95: 64]
    /// @param word1 Word 1 of GBT frame, contains bit [ 63: 32]
    /// @param word0 Word 0 of GBT frame, contains bit [ 31:  0]
    void addGBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0);

    /// Add frame to the container
    /// @param s0hw0l half-word 0 from SAMPA 0 low channel numbers 
    /// @param s0hw1l half-word 1 from SAMPA 0 low channel numbers 
    /// @param s0hw2l half-word 2 from SAMPA 0 low channel numbers 
    /// @param s0hw3l half-word 3 from SAMPA 0 low channel numbers 
    /// @param s0hw0h half-word 0 from SAMPA 0 high channel numbers 
    /// @param s0hw1h half-word 1 from SAMPA 0 high channel numbers 
    /// @param s0hw2h half-word 2 from SAMPA 0 high channel numbers 
    /// @param s0hw3h half-word 3 from SAMPA 0 high channel numbers 
    /// @param s1hw0l half-word 0 from SAMPA 1 low channel numbers 
    /// @param s1hw1l half-word 1 from SAMPA 1 low channel numbers 
    /// @param s1hw2l half-word 2 from SAMPA 1 low channel numbers 
    /// @param s1hw3l half-word 3 from SAMPA 1 low channel numbers 
    /// @param s1hw0h half-word 0 from SAMPA 1 high channel numbers 
    /// @param s1hw1h half-word 1 from SAMPA 1 high channel numbers 
    /// @param s1hw2h half-word 2 from SAMPA 1 high channel numbers 
    /// @param s1hw3h half-word 3 from SAMPA 1 high channel numbers 
    /// @param s2hw0 half-word 0 from SAMPA 2
    /// @param s2hw1 half-word 1 from SAMPA 2
    /// @param s2hw2 half-word 2 from SAMPA 2
    /// @param s2hw3 half-word 3 from SAMPA 2
    /// @param s0adc ADC clock from SAMPA 0
    /// @param s1adc ADC clock from SAMPA 1
    /// @param s2adc ADC clock from SAMPA 2
    /// @param marker additional 16 bit marker which is not part of the actual frame
    void addGBTFrame(short s0hw0l, short s0hw1l, short s0hw2l, short s0hw3l,
                     short s0hw0h, short s0hw1h, short s0hw2h, short s0hw3h,
                     short s1hw0l, short s1hw1l, short s1hw2l, short s1hw3l,
                     short s1hw0h, short s1hw1h, short s1hw2h, short s1hw3h,
                     short s2hw0,  short s2hw1,  short s2hw2,  short s2hw3, 
                     short s0adc,  short s1adc,  short s2adc,  unsigned marker = 0);

//    template<typename... Args> void addGBTFrame(Args&&... args);

    /// Add all frames from file to conatiner
    /// @param fileName Path to file
    void addGBTFramesFromFile(std::string fileName);

    /// Add all frames from file to conatiner
    /// @param fileName Path to file
    /// @param frames Frames to read from file
    void addGBTFramesFromBinaryFile(std::string fileName, int frames = -1);

//    /// Fill output TClonesArray
//    /// @param output Output container
//    void fillOutputContainer(TClonesArray* output);

    /// Set enable the ADC clock warnings
    /// @param val Set it to true or false
    void setEnableAdcClockWarning(bool val)     { mEnableAdcClockWarning = val; };

    /// Set enable the sync pattern position warnings
    /// @param val Set it to true or false
    void setEnableSyncPatternWarning(bool val)  { mEnableSyncPatternWarning = val; };

    /// Set enable compilation of ADC values
    /// @param val Set it to true or false
    void setEnableCompileAdcValues(bool val)    { mEnableCompileAdcValues = val; };

    /// Option to store the inserted GBT frames
    /// @param val Set it to true or false
    void setEnableStoreGBTFrames(bool val)      { mEnableStoreGBTFrames = val; if(!mEnableStoreGBTFrames) mGBTFrames.resize(2);  };

    /// Extracts the digits after all 80 channels were transmitted (5*16)
    /// @param container Digit Container to store the digits in
    /// @return If true, at least one digit was added.
    bool getData(std::vector<DigitData>& container);

    bool getData(std::vector<HalfSAMPAData>& container);

    int getNFramesAnalyzed() const { return mGBTFramesAnalyzed; };

    /// Returns a GBT frame
    /// @param index
    /// @return GBT frame
    GBTFrame getGBTFrame(int index) const { return mGBTFrames[index]; };
    GBTFrame operator[] (int index) const { return mGBTFrames[index]; };
    GBTFrame& operator[] (int index)      { return mGBTFrames[index]; };

    std::vector<GBTFrame>::iterator begin()               { return mGBTFrames.begin(); };
    std::vector<GBTFrame>::const_iterator begin()   const { return mGBTFrames.begin(); };
    std::vector<GBTFrame>::const_iterator cbegin()  const { return mGBTFrames.cbegin(); };
    std::vector<GBTFrame>::iterator end()                 { return mGBTFrames.end(); };
    std::vector<GBTFrame>::const_iterator end()     const { return mGBTFrames.end(); };
    std::vector<GBTFrame>::const_iterator cend()    const { return mGBTFrames.cend(); };

    std::vector<GBTFrame>::reverse_iterator rbegin()               { return mGBTFrames.rbegin(); };
    std::vector<GBTFrame>::const_reverse_iterator rbegin()   const { return mGBTFrames.rbegin(); };
    std::vector<GBTFrame>::const_reverse_iterator crbegin()  const { return mGBTFrames.crbegin(); };
    std::vector<GBTFrame>::reverse_iterator rend()                 { return mGBTFrames.rend(); };
    std::vector<GBTFrame>::const_reverse_iterator rend()     const { return mGBTFrames.rend(); };
    std::vector<GBTFrame>::const_reverse_iterator crend()    const { return mGBTFrames.crend(); };

    /// Sets the timebin
    /// @param val Set to this timebin
    void setTimebin(int val) { mTimebin = val; };

    /// Gets the timebin
    /// @return Timebin for next digits
    int getTimebin() const { return mTimebin; };

    /// Re-Processes all frames after resetting ADC clock and sync Pattern
    void reProcessAllFrames();

    /// Overwrites the ADC clock in all stored frames with a valid sequence
    /// @param sampa ADC clock of this given SAMPA, -1 for all SAMPAs
    /// @param phase Defines which (of the 4 bits) has the rising edge
    void overwriteAdcClock(int sampa, int phase);

  private:
    /// Processes all frames, monitors ADC clock, searches for sync pattern,...
    void processAllFrames();

    /// Processes the last inserted frame, monitors ADC clock, searches for sync pattern,...
    /// @param iFrame GBT Frame to be processed (ordering is important!!)
    void processFrame(std::vector<GBTFrame>::iterator iFrame);

    /// Checks the ADC clock;
    /// @param iFrame GBT Frame to be processed (ordering is important!!)
    void checkAdcClock(std::vector<GBTFrame>::iterator iFrame);

    /// Searches for the synchronization pattern
    /// @param iFrame GBT Frame to be processed (ordering is important!!)
    /// @return Returns the old Position of low bits of SAMPA 0
    int searchSyncPattern(std::vector<GBTFrame>::iterator iFrame);

    // Compiles the ADC values
    /// @param iFrame GBT Frame to be processed (ordering is important!!)
    void compileAdcValues(std::vector<GBTFrame>::iterator iFrame);

    void resetAdcClock();
    void resetSyncPattern();
    void resetAdcValues();

    std::mutex mAdcMutex;

    std::vector<GBTFrame> mGBTFrames;                ///< GBT Frames container
    std::array<AdcClockMonitor,3> mAdcClock;        ///< ADC clock monitor for the 3 SAMPAs
    std::array<SyncPatternMonitor,5> mSyncPattern;  ///< Synchronization pattern monitor for the 5 half SAMPAs
    std::array<short,5> mPositionForHalfSampa;      ///< Start position of data for all 5 half SAMPAs
    std::array<std::queue<short>*,5> mAdcValues;    ///< Vector to buffer the decoded ADC values, one deque per half SAMPA 

    bool mEnableAdcClockWarning;                    ///< enables the ADC clock warnings
    bool mEnableSyncPatternWarning;                 ///< enables the Sync Pattern warnings
    bool mEnableStoreGBTFrames;                     ///< if true, GBT frames are stored
    bool mEnableCompileAdcValues;                   ///<
    int mCRU;                                       ///< CRU ID of the GBT frames
    int mLink;                                      ///< Link ID of the GBT frames
    int mTimebin;                                   ///< Timebin of last digits extraction 
    int mGBTFramesAnalyzed;                         

    std::array<std::array<short,16>,5> mTmpData; 
};

//template<typename... Args>
//inline
//void GBTFrameContainer::addGBTFrame(Args&&... args)
//{
//  if (!mEnableStoreGBTFrames && (mGBTFrames.size() > 1)) {
//    mGBTFrames[0] = mGBTFrames[1];
//    mGBTFrames[1].setData(std::forward<Args>(args)...);
//  } else {
//    mGBTFrames.emplace_back(std::forward<Args>(args)...);
//  }
//  processFrame(mGBTFrames.end()-1);
//};

inline
void GBTFrameContainer::addGBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0) {
  if (!mEnableStoreGBTFrames && (mGBTFrames.size() > 1)) {
    mGBTFrames[0] = mGBTFrames[1];
    mGBTFrames[1].setData(word3, word2, word1, word0);
  } else {
    mGBTFrames.emplace_back(word3, word2, word1, word0);
  }
  processFrame(mGBTFrames.end()-1);
};

inline
void GBTFrameContainer::addGBTFrame(short s0hw0l, short s0hw1l, short s0hw2l, short s0hw3l,
                                    short s0hw0h, short s0hw1h, short s0hw2h, short s0hw3h,
                                    short s1hw0l, short s1hw1l, short s1hw2l, short s1hw3l,
                                    short s1hw0h, short s1hw1h, short s1hw2h, short s1hw3h,
                                    short s2hw0,  short s2hw1,  short s2hw2,  short s2hw3, 
                                    short s0adc,  short s1adc,  short s2adc,  unsigned marker) {
  if (!mEnableStoreGBTFrames && (mGBTFrames.size() > 1)) {
    mGBTFrames[0] = mGBTFrames[1];
    mGBTFrames[1].setData(s0hw0l, s0hw1l, s0hw2l, s0hw3l, s0hw0h, s0hw1h, s0hw2h, s0hw3h,
                          s1hw0l, s1hw1l, s1hw2l, s1hw3l, s1hw0h, s1hw1h, s1hw2h, s1hw3h,
                          s2hw0, s2hw1, s2hw2, s2hw3, s0adc, s1adc, s2adc, marker);
  } else {
    mGBTFrames.emplace_back(s0hw0l, s0hw1l, s0hw2l, s0hw3l, s0hw0h, s0hw1h, s0hw2h, s0hw3h,
                            s1hw0l, s1hw1l, s1hw2l, s1hw3l, s1hw0h, s1hw1h, s1hw2h, s1hw3h,
                            s2hw0, s2hw1, s2hw2, s2hw3, s0adc, s1adc, s2adc, marker);
  }
  processFrame(mGBTFrames.end()-1);
};

inline
void GBTFrameContainer::addGBTFrame(GBTFrame& frame) 
{
  if (!mEnableStoreGBTFrames && (mGBTFrames.size() > 1)) {
    mGBTFrames[0] = mGBTFrames[1];
    mGBTFrames[1] = frame;
  } else {
    mGBTFrames.emplace_back(frame);
  }

  processFrame(mGBTFrames.end()-1);
};


}
}

#endif
