// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
  PixelReader() = default;
  PixelReader(const PixelReader& cluster) = delete;
  virtual ~PixelReader() = default;

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
  void init() override {mIdx=0;}
  Bool_t getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col, Int_t &label) override;
 private:
  const TClonesArray *mDigitArray;
  Int_t mIdx;
};
 
/// \class RawPixelReader
/// \brief RawPixelReader class for the ITS. Feeds raw data to the Cluster Finder
///
class RawPixelReader : public PixelReader {
 public:
  Bool_t getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col, Int_t &label) override; 
};

 
}
}

#endif /* ALICEO2_ITS_PIXELREADER_H */
