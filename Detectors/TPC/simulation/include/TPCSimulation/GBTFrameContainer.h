/// \file GBTFrameContainer.h
/// \brief Container class for the GBT Frames
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_GBTFRAMECONTAINER_H_
#define ALICEO2_TPC_GBTFRAMECONTAINER_H_

#include "TPCSimulation/GBTFrame.h"
#include "TPCSimulation/AdcClockMonitor.h"
#include "TPCSimulation/SyncPatternMonitor.h"
#include "TPCSimulation/Digit.h"
#include "TPCSimulation/SAMPAData.h"
#include <TClonesArray.h>  
#include <vector>
#include <deque>
#include <iterator>
#include "FairLogger.h"

namespace AliceO2{
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
    /// @param amount Forseen amount of GBT frames
    /// @param cru CRU ID
    /// @param link Link ID
    GBTFrameContainer(int amount, int cru, int link);

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
    void addGBTFrame(char s0hw0l, char s0hw1l, char s0hw2l, char s0hw3l,
                     char s0hw0h, char s0hw1h, char s0hw2h, char s0hw3h,
                     char s1hw0l, char s1hw1l, char s1hw2l, char s1hw3l,
                     char s1hw0h, char s1hw1h, char s1hw2h, char s1hw3h,
                     char s2hw0, char s2hw1, char s2hw2, char s2hw3, 
                     char s0adc, char s1adc, char s2adc, unsigned marker = 0);

    /// Fill output TClonesArray
    /// @param output Output container
    void fillOutputContainer(TClonesArray* output);

    /// Set enable the ADC clock warnings
    /// @param val Set it to true or false
    void setEnableAdcClockWarning(bool val) { mEnableAdcClockWarning = val; };

    /// Set enable the sync pattern position warnings
    /// @param val Set it to true or false
    void setEnableSyncPatternWarning(bool val) { mEnableSyncPatternWarning = val; };

    /// Extracts the digits after all 32 channel were transmitted
    /// @param digitContainer Digit Container to store the digits in
    /// @return If true, at least one digit was added.
    bool getDigits(std::vector<Digit> *digitContainer);

    bool getData(std::vector<SAMPAData> *container);

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

    void resetAdcClock();
    void resetSyncPattern();
    void resetAdcValues();


    std::vector<GBTFrame> mGBTFrames;               ///< GBT Frames container
    std::vector<AdcClockMonitor> mAdcClock;         ///< ADC clock monitor for the 3 SAMPAs
    std::vector<SyncPatternMonitor> mSyncPattern;   ///< Synchronization pattern monitor for the 3 SAMPAs
    std::vector<int> mPositionForHalfSampa;         ///< Start position of data for all 5 half SAMPAs
    std::vector<std::deque<int>> mAdcValues;        ///< Vector to buffer the decoded ADC values, one deque per half SAMPA 

    bool mEnableAdcClockWarning;                    ///< enables the ADC clock warnings
    bool mEnableSyncPatternWarning;                 ///< enables the Sync Pattern warnings
    int mCRU;                                       ///< CRU ID of the GBT frames
    int mLink;                                      ///< Link ID of the GBT frames
    int mTimebin;                                   ///< Timebin of last digits extraction 
};
}
}

#endif
