// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataReader.h
/// \brief Definition of the TOF hit reader

#ifndef ALICEO2_TOF_DATAREADER_H
#define ALICEO2_TOF_DATAREADER_H

#include "TOFBase/Digit.h"

namespace o2
{
namespace tof
{
/// \class DataReader
/// \brief DataReader class for TOF
///
class DataReader
{

 public:
  /// Transient data for single strip digits
  struct StripData {
    UShort_t stripID = 0;      // strip id
    std::vector<Digit> digits; // vector of digits

    void clear() { digits.clear(); }
  };

  DataReader() = default;
  DataReader(const DataReader& cluster) = delete;
  virtual ~DataReader() = default;

  DataReader& operator=(const DataReader& src) = delete;

  virtual void init() = 0;

  virtual Bool_t getNextStripData(StripData& stripData) = 0;

  //
 protected:
  //
};

//_______________________________________________________________________

/// \class DigitDataReader
/// \brief DigitDataReader class for TOF. Feeds the MC digits to the Cluster Finder
///
class DigitDataReader : public DataReader
{
 public:
  DigitDataReader() = default;
  void setDigitArray(const std::vector<o2::tof::Digit>* a)
  {
    mDigitArray = a;
    mIdx = 0;
  }

  void init() override
  {
    mIdx = 0;
    mLastDigit = nullptr;
  }

  Bool_t getNextStripData(StripData& stripData) override;

 private:
  const std::vector<o2::tof::Digit>* mDigitArray = nullptr;
  const Digit* mLastDigit = nullptr;
  Int_t mIdx = 0;
};

//_______________________________________________________________________

/// \class RawDataReader
/// \brief RawDataReader class for TOF. Feeds raw data to the Cluster Finder
///
class RawDataReader : public DataReader
{
 public:
  Bool_t getNextStripData(StripData& stripData) override;
};

} // namespace tof
} // namespace o2

#endif /* ALICEO2_TOF_DATAREADER_H */
