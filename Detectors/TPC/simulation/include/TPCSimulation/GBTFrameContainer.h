/// \file GBTFrameContainer.h
/// \brief Container class for the GBT Frames
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_GBTFRAMECONTAINER_H_
#define ALICEO2_TPC_GBTFRAMECONTAINER_H_

#include "TPCSimulation/GBTFrame.h"
#include "TPCSimulation/AdcClockMonitor.h"
#include <TClonesArray.h>  
#include <vector>
#include "FairLogger.h"

namespace AliceO2{
namespace TPC{

/// \class GBTFrameContainer
/// \brief GBT Frame container class

class GBTFrameContainer {
  public:

    /// Default constructor
    GBTFrameContainer();

    /// Constructor
    /// @param amount Forseen amount of GBT frames
    GBTFrameContainer(int amount);

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

  private:
    /// Processes the last inserted frame, monitors ADC clock, searches for sync pattern,...
    void processLastFrame();

    std::vector<GBTFrame> mGBTFrames;   ///< GBT Frames container
    AdcClockMonitor *mAdcClock;       ///< ADC clock monitor classes for the 3 SAMPAs
};

inline
void GBTFrameContainer::reset() 
{
  mGBTFrames.clear();
//  for (auto &aGBTFrame : mGBTFrames) {
//    if (aGBTFrame == nullptr) continue;
//    aGBTFrame->reset();
//  }
}

inline 
int GBTFrameContainer::getNentries() 
{
  int counter = 0;
  for (auto &aGBTFrame : mGBTFrames) {
    ++counter;
  }
  return counter;
}

}
}

#endif
