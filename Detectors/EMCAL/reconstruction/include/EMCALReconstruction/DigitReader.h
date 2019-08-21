// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitReader.h
/// \brief Definition of EMCAL digit reader

#ifndef ALICEO2_EMCAL_DIGITREADER_H
#define ALICEO2_EMCAL_DIGITREADER_H

#include "TTree.h"
#include "DataFormatsEMCAL/Digit.h"

namespace o2
{
namespace emcal
{
/// \class DigitReader
/// \brief DigitReader class for EMCAL. Loads digits and feeds them to clusterizer
///
class DigitReader
{
 public:
  DigitReader() = default;
  ~DigitReader() = default;

  void openInput(const std::string fileName);
  bool readNextEntry();
  void clear();
  const std::vector<o2::emcal::Digit>* getDigitArray() const { return mDigitArray; };

 private:
  std::vector<o2::emcal::Digit>* mDigitArray = nullptr;
  std::unique_ptr<TTree> mInputTree; // input tree for digits
  int mCurrentEntry;                 // current entry in input file

  ClassDef(DigitReader, 1);
};

} // namespace emcal
} // namespace o2

#endif /* ALICEO2_EMCAL_DIGITREADER_H */
