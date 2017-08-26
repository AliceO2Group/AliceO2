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
#ifndef ALICEO2_ITSMFT_PIXELREADER_H
#define ALICEO2_ITSMFT_PIXELREADER_H

#include <Rtypes.h>
#include "ITSMFTBase/Digit.h"

class TClonesArray;

namespace o2
{
namespace ITSMFT
{
 
  
/// \class PixelReader
/// \brief PixelReader class for the ITSMFT
///
class PixelReader {
  
  public:

  /// Transient data for single fired pixel
  struct PixelData {
    UShort_t row;
    UShort_t col;  
    Int_t labels[Digit::maxLabels] = {-1,-1,-1};
    
    PixelData(const Digit* dig) :
    row(dig->getRow()), col(dig->getColumn())
    {
      for (int i=Digit::maxLabels;i--;) labels[i]=dig->getLabel(i);
    }
    PixelData(UShort_t r, UShort_t c) : row(r), col(c) {}
  };

  /// Transient data for single chip fired pixeld
  struct ChipPixelData {
    UShort_t chipID = 0;               // chip id within detector
    UInt_t   roFrame = 0;              // readout frame ID
    Double_t timeStamp = 0.;                 // Fair time ?
    std::vector<PixelData> pixels;     // vector of pixeld
    
    void clear() {
      pixels.clear();
    }
  };
  
  PixelReader() = default;
  PixelReader(const PixelReader& cluster) = delete;
  virtual ~PixelReader() = default;

  PixelReader& operator=(const PixelReader& src) = delete;

  virtual void init() = 0;
  virtual Bool_t getNextChipData(ChipPixelData &chipData) = 0;
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

  void init() override {
    mIdx=0;
    mLastDigit = nullptr;
  }
  
  Bool_t getNextChipData(ChipPixelData &chipData) override;

 private:

  void addPixel(PixelReader::ChipPixelData &chipData, const Digit* dig) {
    // add new fired pixel
    chipData.pixels.emplace_back(dig);
  }
  
  const TClonesArray *mDigitArray;
  const Digit* mLastDigit = nullptr;
  Int_t        mIdx = 0;
};
 
/// \class RawPixelReader
/// \brief RawPixelReader class for the ITS. Feeds raw data to the Cluster Finder
///
class RawPixelReader : public PixelReader {
 public:
  Bool_t getNextChipData(ChipPixelData &chipData) override;
};


}
}

#endif /* ALICEO2_ITS_PIXELREADER_H */
