/// \file PixelReader.h
/// \brief Definition of the ITS pixel reader
#ifndef ALICEO2_ITS_PIXELREADER_H
#define ALICEO2_ITS_PIXELREADER_H

#include <Rtypes.h>

namespace o2
{
namespace ITS
{
/// \class PixelReader
/// \brief PixelReader class for the ITS
///
class PixelReader {
 public:
  PixelReader();
  PixelReader(const PixelReader& cluster) = delete;
  ~PixelReader() {};

  PixelReader& operator=(const PixelReader& cluster) = delete;

  virtual Bool_t getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col) = 0; 
  //
 protected:
  //

};

/// \class DigitPixelReader
/// \brief DigitPixelReader class for the ITS. Feeds the MC digits to the Cluster Finder
///
class DigitPixelReader : public PixelReader {
 public:
  virtual Bool_t getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col); 
};
 
/// \class RawPixelReader
/// \brief RawPixelReader class for the ITS. Feeds raw data to the Cluster Finder
///
class RawPixelReader : public PixelReader {
 public:
  virtual Bool_t getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col); 
};

 
}
}

#endif /* ALICEO2_ITS_PIXELREADER_H */
