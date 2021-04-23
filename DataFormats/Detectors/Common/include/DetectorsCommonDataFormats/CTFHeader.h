// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CTFHeader.h
/// \brief Header for CTF collection

#ifndef ALICEO2_CTF_HEADER_H
#define ALICEO2_CTF_HEADER_H

#include <Rtypes.h>
#include <string>
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace ctf
{

struct CTFHeader {

  uint64_t run;                           // run number
  uint32_t firstTForbit = 0;              // first orbit of time frame as unique identifier within the run
  o2::detectors::DetID::mask_t detectors; // mask of represented detectors

  std::string describe() const;
  void print() const;

  ClassDefNV(CTFHeader, 2)
};

std::ostream& operator<<(std::ostream& stream, const CTFHeader& c);

} // namespace ctf
} // namespace o2

#endif
