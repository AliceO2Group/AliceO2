// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file OrbitInfo.h
 * C++ Muon MCH class that holds information about processed orbits
 * @author Andrea Ferrero
 */

#ifndef ALICEO2_MCH_BASE_ORBITINFO_H_
#define ALICEO2_MCH_BASE_ORBITINFO_H_

#include <gsl/span>

#include "Rtypes.h"

namespace o2
{
namespace mch
{

// \class OrbitInfo
/// \brief MCH orbit info implementation
class OrbitInfo
{
 public:
  OrbitInfo() = default;

  OrbitInfo(gsl::span<const std::byte> rdhBuffer);
  ~OrbitInfo() = default;

  uint64_t get() { return mOrbitInfo; }
  uint32_t getOrbit() const { return (mOrbitInfo & 0xFFFFFFFF); }
  uint8_t getLinkID() const { return ((mOrbitInfo >> 32) & 0xFF); }
  uint16_t getFeeID() const { return ((mOrbitInfo >> 40) & 0xFF); }

  friend bool operator==(const OrbitInfo& o1, const OrbitInfo& o2);
  friend bool operator!=(const OrbitInfo& o1, const OrbitInfo& o2);

 private:
  uint64_t mOrbitInfo = {0};

  ClassDefNV(OrbitInfo, 1);
}; //class OrbitInfo

bool operator==(const OrbitInfo& o1, const OrbitInfo& o2);
bool operator!=(const OrbitInfo& o1, const OrbitInfo& o2);

} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_BASE_ORBITINFO_H_
