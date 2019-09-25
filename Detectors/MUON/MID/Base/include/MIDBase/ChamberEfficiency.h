// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/ChamberEfficiency.h
/// \brief  Measured values of the RPC efficiency
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   03 March 2019
#ifndef O2_MID_CHAMBEREFFICIENCY_H
#define O2_MID_CHAMBEREFFICIENCY_H

#include <map>
#include <array>

namespace o2
{
namespace mid
{
class ChamberEfficiency
{
 public:
  enum class EffType {
    BendPlane,
    NonBendPlane,
    BothPlanes
  };

  double getEfficiency(int deId, int columnId, int line, EffType type) const;

  // void setEfficiency(uint32_t nPassed, uint32_t nTotal, int deId, int columnId, int line, EffType type);
  void addEntry(bool isEfficientBP, bool isEfficientNBP, int deId, int columnId, int line);

 private:
  int typeToIdx(EffType type) const;
  int indexToInt(int deId, int columnId, int line) const;
  std::map<int, std::array<uint32_t, 4>> mCounters; ///< Efficiency counters
};

ChamberEfficiency createDefaultChamberEfficiency();

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHAMBEREFFICIENCY_H */
