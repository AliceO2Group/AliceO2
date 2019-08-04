// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GBTFrame.h
/// \brief GBT Frame object
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_GBTFRAME_H_
#define ALICEO2_TPC_GBTFRAME_H_

#include <iosfwd>
#include <iomanip>
#include <vector>
#include <array>

namespace o2
{
namespace tpc
{

#define BIT00(x) ((x & 0x1) >> 0)
#define BIT01(x) ((x & 0x2) >> 1)
#define BIT02(x) ((x & 0x4) >> 2)
#define BIT03(x) ((x & 0x8) >> 3)
#define BIT04(x) ((x & 0x10) >> 4)
#define BIT05(x) ((x & 0x20) >> 5)
#define BIT06(x) ((x & 0x40) >> 6)
#define BIT07(x) ((x & 0x80) >> 7)
#define BIT08(x) ((x & 0x100) >> 8)
#define BIT09(x) ((x & 0x200) >> 9)
#define BIT10(x) ((x & 0x400) >> 10)
#define BIT11(x) ((x & 0x800) >> 11)
#define BIT12(x) ((x & 0x1000) >> 12)
#define BIT13(x) ((x & 0x2000) >> 13)
#define BIT14(x) ((x & 0x4000) >> 14)
#define BIT15(x) ((x & 0x8000) >> 15)
#define BIT16(x) ((x & 0x10000) >> 16)
#define BIT17(x) ((x & 0x20000) >> 17)
#define BIT18(x) ((x & 0x40000) >> 18)
#define BIT19(x) ((x & 0x80000) >> 19)
#define BIT20(x) ((x & 0x100000) >> 20)
#define BIT21(x) ((x & 0x200000) >> 21)
#define BIT22(x) ((x & 0x400000) >> 22)
#define BIT23(x) ((x & 0x800000) >> 23)
#define BIT24(x) ((x & 0x1000000) >> 24)
#define BIT25(x) ((x & 0x2000000) >> 25)
#define BIT26(x) ((x & 0x4000000) >> 26)
#define BIT27(x) ((x & 0x8000000) >> 27)
#define BIT28(x) ((x & 0x10000000) >> 28)
#define BIT29(x) ((x & 0x20000000) >> 29)
#define BIT30(x) ((x & 0x40000000) >> 30)
#define BIT31(x) ((x & 0x80000000) >> 31)

/// \class GBTFrame
/// \brief GBTFrame class for the TPC
class GBTFrame
{
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
  ~GBTFrame() = default;

  /// Get the marker of the frame
  /// @return marker of the frame
  short getMarker() const { return (mWords[3] >> 16) & 0xFFFF; };

  /// Get a half-word of a SAMPA chip (5 bit of data)
  /// @param sampa half-word this SAMPA chip (0-2, 3=0, 4=1)
  /// @param halfword this half-word of the SAMPA (4 are included in GBT frame)
  /// @param chan which channels, 0 for channel 0-15, 1 for channel 16-31, ignored for SAMPA 2
  /// @return requested half-word
  short getHalfWord(const short sampa, const short halfword, const short chan = 0) const { return mHalfWords[sampa][chan][halfword]; };

  /// Get ADC sampling clock of a SAMPA chip (4 time bits)
  /// @param sampa ADC clock of this SAMPA chip (0-2, 3=0, 4=1)
  /// @return requested ADC sampling clock bits
  short getAdcClock(short sampa) const { return mAdcClock[sampa]; };

  /// Set Adc sampling clock of a SAMPA chip
  /// @param sampa ADC clock of this SAMPA chip (0-2, 3=0, 4=1), -1 for all SAMPAs
  /// @param clock 4 sampling clock bits
  void setAdcClock(int sampa, int clock);

  ///
  /// Set data
  /// @param other Other GBT Frame
  //    void setData(const GBTFrame& other);

  /// Set data
  /// @param word3 Word 3 of GBT frame, contains bit [127: 96], [127:112] are not part of the actual frame
  /// @param word2 Word 2 of GBT frame, contains bit [ 95: 64]
  /// @param word1 Word 1 of GBT frame, contains bit [ 63: 32]
  /// @param word0 Word 0 of GBT frame, contains bit [ 31:  0]
  void setData(unsigned word3, unsigned word2, unsigned word1, unsigned word0);

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
  void setData(short s0hw0l, short s0hw1l, short s0hw2l, short s0hw3l,
               short s0hw0h, short s0hw1h, short s0hw2h, short s0hw3h,
               short s1hw0l, short s1hw1l, short s1hw2l, short s1hw3l,
               short s1hw0h, short s1hw1h, short s1hw2h, short s1hw3h,
               short s2hw0, short s2hw1, short s2hw2, short s2hw3,
               short s0adc, short s1adc, short s2adc, unsigned marker = 0);

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
  friend std::ostream& operator<<(std::ostream& out, const GBTFrame& f) { return f.Print(out); }

 private:
  void calculateHalfWords();
  void calculateAdcClock();

  std::array<unsigned, 4> mWords;
  // Word 3 of GBT frame contains bits [127: 96], [127:112] are reserved for marker
  // Word 2 of GBT frame contains bits [ 95: 64]
  // Word 1 of GBT frame contains bits [ 63: 32]
  // Word 0 of GBT frame contains bits [ 31:  0]

  std::array<std::array<std::array<short, 4>, 2>, 3> mHalfWords;
  //                           halfWord
  //                               channels (low or high)
  //                                   sampa

  std::array<short, 3> mAdcClock;
};

inline void GBTFrame::calculateHalfWords()
{

  mHalfWords[0][0][0] = (BIT19(mWords[0]) << 4) | (BIT15(mWords[0]) << 3) | (BIT11(mWords[0]) << 2) | (BIT07(mWords[0]) << 1) | BIT03(mWords[0]);
  mHalfWords[0][0][1] = (BIT18(mWords[0]) << 4) | (BIT14(mWords[0]) << 3) | (BIT10(mWords[0]) << 2) | (BIT06(mWords[0]) << 1) | BIT02(mWords[0]);
  mHalfWords[0][0][2] = (BIT17(mWords[0]) << 4) | (BIT13(mWords[0]) << 3) | (BIT09(mWords[0]) << 2) | (BIT05(mWords[0]) << 1) | BIT01(mWords[0]);
  mHalfWords[0][0][3] = (BIT16(mWords[0]) << 4) | (BIT12(mWords[0]) << 3) | (BIT08(mWords[0]) << 2) | (BIT04(mWords[0]) << 1) | BIT00(mWords[0]);

  mHalfWords[0][1][0] = (BIT07(mWords[1]) << 4) | (BIT03(mWords[1]) << 3) | (BIT31(mWords[0]) << 2) | (BIT27(mWords[0]) << 1) | BIT23(mWords[0]);
  mHalfWords[0][1][1] = (BIT06(mWords[1]) << 4) | (BIT02(mWords[1]) << 3) | (BIT30(mWords[0]) << 2) | (BIT26(mWords[0]) << 1) | BIT22(mWords[0]);
  mHalfWords[0][1][2] = (BIT05(mWords[1]) << 4) | (BIT01(mWords[1]) << 3) | (BIT29(mWords[0]) << 2) | (BIT25(mWords[0]) << 1) | BIT21(mWords[0]);
  mHalfWords[0][1][3] = (BIT04(mWords[1]) << 4) | (BIT00(mWords[1]) << 3) | (BIT28(mWords[0]) << 2) | (BIT24(mWords[0]) << 1) | BIT20(mWords[0]);

  mHalfWords[1][0][0] = (BIT31(mWords[1]) << 4) | (BIT27(mWords[1]) << 3) | (BIT23(mWords[1]) << 2) | (BIT19(mWords[1]) << 1) | BIT15(mWords[1]);
  mHalfWords[1][0][1] = (BIT30(mWords[1]) << 4) | (BIT26(mWords[1]) << 3) | (BIT22(mWords[1]) << 2) | (BIT18(mWords[1]) << 1) | BIT14(mWords[1]);
  mHalfWords[1][0][2] = (BIT29(mWords[1]) << 4) | (BIT25(mWords[1]) << 3) | (BIT21(mWords[1]) << 2) | (BIT17(mWords[1]) << 1) | BIT13(mWords[1]);
  mHalfWords[1][0][3] = (BIT28(mWords[1]) << 4) | (BIT24(mWords[1]) << 3) | (BIT20(mWords[1]) << 2) | (BIT16(mWords[1]) << 1) | BIT12(mWords[1]);

  mHalfWords[1][1][0] = (BIT19(mWords[2]) << 4) | (BIT15(mWords[2]) << 3) | (BIT11(mWords[2]) << 2) | (BIT07(mWords[2]) << 1) | BIT03(mWords[2]);
  mHalfWords[1][1][1] = (BIT18(mWords[2]) << 4) | (BIT14(mWords[2]) << 3) | (BIT10(mWords[2]) << 2) | (BIT06(mWords[2]) << 1) | BIT02(mWords[2]);
  mHalfWords[1][1][2] = (BIT17(mWords[2]) << 4) | (BIT13(mWords[2]) << 3) | (BIT09(mWords[2]) << 2) | (BIT05(mWords[2]) << 1) | BIT01(mWords[2]);
  mHalfWords[1][1][3] = (BIT16(mWords[2]) << 4) | (BIT12(mWords[2]) << 3) | (BIT08(mWords[2]) << 2) | (BIT04(mWords[2]) << 1) | BIT00(mWords[2]);

  mHalfWords[2][0][0] = (BIT11(mWords[3]) << 4) | (BIT07(mWords[3]) << 3) | (BIT03(mWords[3]) << 2) | (BIT31(mWords[2]) << 1) | BIT27(mWords[2]);
  mHalfWords[2][0][1] = (BIT10(mWords[3]) << 4) | (BIT06(mWords[3]) << 3) | (BIT02(mWords[3]) << 2) | (BIT30(mWords[2]) << 1) | BIT26(mWords[2]);
  mHalfWords[2][0][2] = (BIT09(mWords[3]) << 4) | (BIT05(mWords[3]) << 3) | (BIT01(mWords[3]) << 2) | (BIT29(mWords[2]) << 1) | BIT25(mWords[2]);
  mHalfWords[2][0][3] = (BIT08(mWords[3]) << 4) | (BIT04(mWords[3]) << 3) | (BIT00(mWords[3]) << 2) | (BIT28(mWords[2]) << 1) | BIT24(mWords[2]);

  //  mHalfWords[2][1][0] = mHalfWords[2][0][0];
  //  mHalfWords[2][1][1] = mHalfWords[2][0][1];
  //  mHalfWords[2][1][2] = mHalfWords[2][0][2];
  //  mHalfWords[2][1][3] = mHalfWords[2][0][3];

  calculateAdcClock();
};

inline void GBTFrame::calculateAdcClock()
{
  mAdcClock[0] = (mWords[1] >> 8) & 0xF;
  mAdcClock[1] = (mWords[2] >> 20) & 0xF;
  mAdcClock[2] = (mWords[3] >> 12) & 0xF;
};

inline void GBTFrame::setData(unsigned word3, unsigned word2, unsigned word1, unsigned word0)
{
  mWords[3] = word3;
  mWords[2] = word2;
  mWords[1] = word1;
  mWords[0] = word0;

  calculateHalfWords();
};

inline GBTFrame::GBTFrame()
  : GBTFrame(0, 0, 0, 0){};

inline GBTFrame::GBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0)
{
  mWords[3] = word3;
  mWords[2] = word2;
  mWords[1] = word1;
  mWords[0] = word0;

  calculateHalfWords();
};

inline GBTFrame::GBTFrame(const GBTFrame& other) = default;

inline void GBTFrame::getGBTFrame(unsigned& word3, unsigned& word2, unsigned& word1, unsigned& word0) const
{
  word3 = mWords[3];
  word2 = mWords[2];
  word1 = mWords[1];
  word0 = mWords[0];
};

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_GBTFRAME_H_
