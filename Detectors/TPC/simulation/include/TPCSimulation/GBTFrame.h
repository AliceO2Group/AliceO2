/// \file GBTFrame.h
/// \brief GBT Frame object
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_GBTFRAME_H_
#define ALICEO2_TPC_GBTFRAME_H_

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>

namespace AliceO2 {
namespace TPC {

/// \class GBTFrame
/// \brief GBTFrame class for the TPC
class GBTFrame {
  public:

    /// Default Constructor
    GBTFrame();

    /// Constructor
    /// @param word3 Word 3 of GBT frame, contains bit [127: 96], [127:112] are not part of the actual frame
    /// @param word2 Word 2 of GBT frame, contains bit [ 95: 64]
    /// @param word1 Word 1 of GBT frame, contains bit [ 63: 32]
    /// @param word0 Word 0 of GBT frame, contains bit [ 31:  0]
    GBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0);

    /// Constructor
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
    GBTFrame(short s0hw0l, short s0hw1l, short s0hw2l, short s0hw3l,
             short s0hw0h, short s0hw1h, short s0hw2h, short s0hw3h,
             short s1hw0l, short s1hw1l, short s1hw2l, short s1hw3l,
             short s1hw0h, short s1hw1h, short s1hw2h, short s1hw3h,
             short s2hw0, short s2hw1, short s2hw2, short s2hw3, 
             short s0adc, short s1adc, short s2adc, unsigned marker = 0);

    /// Copy Contructor
    /// @param other GBT Frame to be copied
    GBTFrame(const GBTFrame& other);

    /// Destructor
    ~GBTFrame();

    /// Get the marker of the frame
    /// @return marker of the frame
    short getMarker() const { return (mWords[3] >> 16) & 0xFFFF; };

    /// Get a half-word of a SAMPA chip (5 bit of data)
    /// @param sampa half-word this SAMPA chip (0-2, 3=0, 4=1)
    /// @param halfword this half-word of the SAMPA (4 are included in GBT frame)
    /// @param chan which channels, 0 for channel 0-15, 1 for channel 16-31, ignored for SAMPA 2
    /// @return requested half-word
    const short& getHalfWord(const short& sampa, const short& halfword, const short& chan = 0) const { return mHalfWords[sampa][chan][halfword]; };

    /// Get ADC sampling clock of a SAMPA chip (4 time bits)
    /// @param sampa ADC clock of this SAMPA chip (0-2, 3=0, 4=1)
    /// @return requested ADC sampling clock bits
    const short& getAdcClock(short sampa) const { return mAdcClock[sampa]; };

    /// Set Adc sampling clock of a SAMPA chip
    /// @param sampa ADC clock of this SAMPA chip (0-2, 3=0, 4=1), -1 for all SAMPAs 
    /// @param clock 4 sampling clock bits
    void setAdcClock(int sampa, int clock);

    ///
    /// Set data
    /// @param other Other GBT Frame
    void setData(const GBTFrame& other);

    /// Set data
    /// @param word3 Word 3 of GBT frame, contains bit [127: 96], [127:112] are not part of the actual frame
    /// @param word2 Word 2 of GBT frame, contains bit [ 95: 64]
    /// @param word1 Word 1 of GBT frame, contains bit [ 63: 32]
    /// @param word0 Word 0 of GBT frame, contains bit [ 31:  0]
    void setData(unsigned& word3, unsigned& word2, unsigned& word1, unsigned& word0);

    /// Set Data
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
    void setData(short& s0hw0l, short& s0hw1l, short& s0hw2l, short& s0hw3l,
                 short& s0hw0h, short& s0hw1h, short& s0hw2h, short& s0hw3h,
                 short& s1hw0l, short& s1hw1l, short& s1hw2l, short& s1hw3l,
                 short& s1hw0h, short& s1hw1h, short& s1hw2h, short& s1hw3h,
                 short& s2hw0,  short& s2hw1,  short& s2hw2,  short& s2hw3, 
                 short& s0adc,  short& s1adc,  short& s2adc,  unsigned marker = 0);

    /// Get the GBT frame
    /// @param word3 bit [127: 96] of GBT frame is written to this
    /// @param word2 bit [ 95: 64] of GBT frame is written to this
    /// @param word1 bit [ 63: 32] of GBT frame is written to this
    /// @param word0 bit [ 31:  0] of GBT frame is written to this
    void getGBTFrame(unsigned& word3, unsigned& word2, unsigned& word1, unsigned& word0) const;

    /// Print function: Print GBT frame on the output stream
    /// @param output Stream to put the GBT frame on
    /// @return The output stream
    std::ostream& Print(std::ostream& output) const; 
    friend std::ostream& operator<< (std::ostream& out, const GBTFrame& f) { return f.Print(out); }

  private:

    bool getBit(unsigned word, unsigned lsb) const;
    short getBits(short word, unsigned width, unsigned lsb) const;
    unsigned getBits(unsigned word, unsigned width, unsigned lsb) const;
    unsigned combineBits(std::vector<bool> bits) const;
    short combineBitsOfFrame(short bit0, short bit1, short bit2, short bit3, short bit4) const;
    void calculateHalfWords();
    void calculateAdcClock();

    std::array<unsigned, 4> mWords;
                // Word 3 of GBT frame contains bits [127: 96], [127:112] are reserved for marker
                // Word 2 of GBT frame contains bits [ 95: 64]
                // Word 1 of GBT frame contains bits [ 63: 32]
                // Word 0 of GBT frame contains bits [ 31:  0]
                
    std::array<std::array<std::array<short,  5>, 2>, 3> mHalfWords;
                //                          halfWord
                //                              channels (low or high)
                //                                  sampa

    std::array<short, 3> mAdcClock;
};
}
}

#endif // ALICEO2_TPC_GBTFRAME_H_
