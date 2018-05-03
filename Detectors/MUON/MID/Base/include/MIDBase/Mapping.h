// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/Mapping.h
/// \brief  Mapping for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 December 2017
#ifndef O2_MID_MAPPING_H
#define O2_MID_MAPPING_H

#include <array>
#include <vector>
#include "MIDBase/MpArea.h"

namespace o2
{
namespace mid
{
class Mapping
{
 public:
  Mapping();
  virtual ~Mapping() = default;

  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  Mapping(Mapping&&) = delete;
  Mapping& operator=(Mapping&&) = delete;

  /// Indexes required to define a strip in the detection element
  struct MpStripIndex {
    /// Check if Strip is Valid
    bool isValid() { return (column < 7 && line < 4 && strip < 16); }
    int column; /// Column in the DE
    int line;   /// Line of the local board in the column
    int strip;  /// Stip in the local board
  };

  MpArea stripByLocation(int strip, int cathode, int line, int column, int deId, bool warn = true) const;
  MpArea stripByLocationInBoard(int strip, int cathode, int boardId, int chamber, bool warn = true) const;
  // MpArea stripByPosition(double xPos, double yPos, int cathode, int deId, bool warn = true) const;
  MpStripIndex stripByPosition(double xPos, double yPos, int cathode, int deId, bool warn = true) const;

  double getStripSize(int strip, int cathode, int column, int deId) const;
  int getNStripsNBP(int column, int deId) const;
  int getFirstColumn(int deId) const;
  int getFirstBoardBP(int column, int deId) const;
  int getLastBoardBP(int column, int deId) const;
  int getBoardId(int line, int column, int deId, bool warn = true) const;
  std::vector<MpStripIndex> getNeighbours(const Mapping::MpStripIndex& stripIndex, int cathode, int deId) const;
  bool isValid(int deId, int column, int cathode = 0, int line = 0, int strip = 0) const;

 private:
  /// Structure of a column in a DE in the internal mapping
  struct MpColumn {
    int nStripsNBP;
    int stripPitchNBP;
    int stripPitchBP;
    std::vector<int> boardsBP;
  };

  /// Detection Element structure in the internal mapping
  struct MpDE {
    std::array<MpColumn, 7> columns; ///< columns in DE
  };

  /// Indexes required to define a local board in the Detection Element
  struct MpBoardIndex {
    int deType; ///< Id of the corresponding detection elememt type
    int column; ///< Column in the DE
    int line;   ///< Line in the DE
  };

  void init();
  void setupSegmentation(int rpcType, int column, int nStripsNBP, int stripPitchNBP, int nBoardsBP, int firstBoardId,
                         bool isBelowBeamPipe = false);
  void setupSegmentationLastColumn(int rpcType, int boardId);
  void buildDETypeLarge(int rpcType, std::array<int, 7> boards);
  void buildDETypeMedium(int rpcType, std::array<int, 7> boards, bool largeNonBend);
  void buildDETypeCut(int rpcType, std::array<int, 7> boards, bool isBelowBeamPipe);
  void buildDETypeShort(int rpcType, std::array<int, 7> boards);
  /// Gets the RPC type of detection element deId
  inline int getRPCType(int deId) const { return deId % 9; }
  double getStripSize(int chamber, int stripPitch, int strip = 0) const;
  double getColumnLeftPosition(int column, int chamber, int rpcType) const;
  double getColumnBottomPosition(int column, int chamber, int rpcType) const;
  double getColumnHeight(int column, int chamber, int rpcType) const;
  double getColumnWidth(int column, int chamber, int rpcType) const;
  double getStripLowEdge(int strip, int stripPitch, int line, int chamber) const;
  double getStripLeftEdge(int strip, int stripPitch, int column, int chamber, int rpcType) const;
  int getColumn(double xPos, int chamber, int rpcType) const;
  int getLine(double yPos, const MpColumn& column, int chamber) const;
  bool isValidColumn(int column, int rpcType) const;
  bool isValidLine(int line, int column, int rpcType) const;
  std::vector<MpStripIndex> getNeighboursBP(const MpStripIndex& stripIndex, int rpcType) const;
  std::vector<MpStripIndex> getNeighboursNBP(const MpStripIndex& stripIndex, int rpcType) const;

  std::array<MpDE, 9> mDetectionElements;      ///< Array of detection element
  std::array<MpBoardIndex, 118> mBoardIndexes; ///< Array of board indexes
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_MAPPING_H */
