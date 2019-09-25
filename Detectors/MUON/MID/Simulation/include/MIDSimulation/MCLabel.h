// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/MCLabel.h
/// \brief  Label for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   27 February 2019

#ifndef O2_MID_MCLABEL_H
#define O2_MID_MCLABEL_H

#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace mid
{
class MCLabel : public o2::MCCompLabel
{
 private:
  uint32_t mStripsInfo = 0; /// Strip information

  static constexpr unsigned int sMaskDE = 0x7F;
  static constexpr unsigned int sMaskColumn = 0x7;
  static constexpr unsigned int sMaskCathode = 0x1;
  static constexpr unsigned int sMaskStrip = 0x3F;

  static constexpr unsigned int sOffsetDE = 0;
  static constexpr unsigned int sOffsetColumn = 7;
  static constexpr unsigned int sOffsetCathode = 10;
  static constexpr unsigned int sOffsetStripFirst = 11;
  static constexpr unsigned int sOffsetStripLast = 17;

  void set(int value, unsigned int mask, unsigned int offset);
  int get(unsigned int mask, unsigned int offset) const { return (mStripsInfo >> offset) & mask; }

 public:
  MCLabel() = default;
  MCLabel(int trackID, int eventID, int srcID, int deId, int columnId, int cathode, int firstStrip, int lastStrip);

  bool operator==(const MCLabel& other) const;

  /// Sets the detection element ID
  void setDEId(int deId) { set(deId, sMaskDE, sOffsetDE); }
  /// Gets the detection element ID
  int getDEId() const { return get(sMaskDE, sOffsetDE); }

  /// Sets the column ID
  void setColumnId(int columnId) { set(columnId, sMaskColumn, sOffsetColumn); }
  /// Gets the column ID
  int getColumnId() const { return get(sMaskColumn, sOffsetColumn); }

  /// Sets the cathode
  void setCathode(int cathode) { set(cathode, sMaskCathode, sOffsetCathode); }
  /// Gets the Cathode
  int getCathode() const { return get(sMaskCathode, sOffsetCathode); }

  /// Sets the first strip
  void setFirstStrip(int firstStrip) { set(firstStrip, sMaskStrip, sOffsetStripFirst); }
  /// Gets the first strip
  int getFirstStrip() const { return get(sMaskStrip, sOffsetStripFirst); }

  /// Sets the last strip
  void setLastStrip(int lastStrip) { set(lastStrip, sMaskStrip, sOffsetStripLast); }
  /// Gets the last strip
  int getLastStrip() const { return get(sMaskStrip, sOffsetStripLast); }

  /// Gets the line number
  static int getLine(int strip) { return strip / 4; }
  /// Gets the strip number in line
  static int getStripInLine(int strip) { return strip % 4; }
  /// Gets the strip
  static int getStrip(int strip, int line) { return 16 * line + strip; }

  ClassDefNV(MCLabel, 1);
};
} // namespace mid
} // namespace o2

#endif
