/// \file PixelReader.h
/// \brief Definition of the ITS pixel reader
#ifndef ALICEO2_ITS_PIXELREADER_H
#define ALICEO2_ITS_PIXELREADER_H

#include <Rtypes.h>

class TClonesArray;

namespace o2
{
namespace ITS
{
/// \class PixelReader
/// \brief PixelReader class for the ITS
///
class PixelReader {
 public:
  PixelReader() {}
  PixelReader(const PixelReader& cluster) = delete;
  ~PixelReader() {};

  PixelReader& operator=(const PixelReader& cluster) = delete;

  virtual void init() = 0;
  virtual Bool_t getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col, Int_t &label) = 0; 
  //
 protected:
  //

};

/// \class DigitPixelReader
/// \brief DigitPixelReader class for the ITS. Feeds the MC digits to the Cluster Finder
///
class DigitPixelReader : public PixelReader {
 public:
  DigitPixelReader() : mDigitArray(nullptr), mIdx(0) {}
  void setDigitArray(const TClonesArray *a) { mDigitArray=a; mIdx=0; }
  void init() {mIdx=0;}
  virtual Bool_t getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col, Int_t &label);
 private:
  const TClonesArray *mDigitArray;
  Int_t mIdx;
};
 
/// \class RawPixelReader
/// \brief RawPixelReader class for the ITS. Feeds raw data to the Cluster Finder
///
class RawPixelReader : public PixelReader {
 public:
  virtual Bool_t getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col, Int_t &label); 
};

 
}
}

#endif /* ALICEO2_ITS_PIXELREADER_H */
