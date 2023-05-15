// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/GlobalMapping/src/GlobalMapper.cxx
/// \brief  Global mapper for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 April 2023

#include "MIDGlobalMapping/GlobalMapper.h"

#include <fmt/format.h>
#include "MIDBase/DetectorParameters.h"
#include "MIDBase/GeometryParameters.h"

namespace o2
{
namespace mid
{
int getStripId(int deId, int columnId, int lineId, int stripId, int cathode)
{
  return stripId | (cathode << 4) | (lineId << 5) | (columnId << 7) | (deId << 10);
}

std::array<int, 4> GlobalMapper::getStripGeom(int deId, int columnId, int lineId, int stripId, int cathode) const
{
  int irpc = detparams::getRPCLine(deId);
  int offsetNB = 32 * columnId;
  int offsetB = 32 * (2 * irpc - detparams::NRPCLines);
  int xpos = offsetNB;
  int ypos = offsetB;
  int pitch = static_cast<int>(mMapping.getStripSize(stripId, cathode, columnId, deId));
  int xwidth = pitch;
  int ywidth = pitch;

  if (cathode == 0) {
    ypos += pitch * (16 * lineId + stripId);
    xwidth = 32;
    if (columnId == 6) {
      xwidth += 16;
    }
  } else {
    xpos += pitch * stripId;
    int firstLine = mMapping.getFirstBoardBP(columnId, deId);
    int lastLine = mMapping.getLastBoardBP(columnId, deId);
    int nLines = lastLine - firstLine + 1;
    ypos += 16 * firstLine;
    ywidth = (nLines == 3) ? 48 : 64;
    if (columnId == 6 && stripId >= 8) {
      // The strip pitch for the previous strips was 4 cm
      // while the one for this strip is 2 cm
      // So we need to add 2 * 8 to the offset...
      xpos += 8 * pitch;
    }
  }

  if (geoparams::isShortRPC(deId) && columnId == 1) {
    xpos += 16;
    if (cathode == 0) {
      xwidth = 16;
    }
  }

  if (!detparams::isRightSide(deId)) {
    xpos *= -1;
    xwidth *= -1;
  }

  return {mScaleFactor * xpos, mScaleFactor * ypos, mScaleFactor * xwidth, mScaleFactor * ywidth};
}

ExtendedMappingInfo GlobalMapper::buildExtendedInfo(int deId, int columnId, int lineId, int stripId, int cathode) const
{
  ExtendedMappingInfo info;
  info.id = getStripId(deId, columnId, lineId, stripId, cathode);
  auto locId = static_cast<int>(mCrateMapper.deLocalBoardToRO(deId, columnId, lineId));
  info.locId = locId;
  std::string side = detparams::isRightSide(deId) ? "R" : "L";
  auto crateId = (locId >> 4) % 8;
  auto locInCrate = (locId & 0xF);
  info.rpc = detparams::getDEName(deId);
  info.deId = deId;
  info.columnId = columnId;
  info.lineId = lineId;
  info.stripId = stripId;
  info.cathode = cathode;
  info.locIdDcs = fmt::format("{}{}{}{}", crateId, side, (locInCrate >= 8 ? "1" : "0"), locInCrate);
  auto geom = getStripGeom(deId, columnId, lineId, stripId, cathode);
  info.xpos = geom[0];
  info.ypos = geom[1];
  info.xwidth = geom[2];
  info.ywidth = geom[3];
  return info;
}

std::vector<ExtendedMappingInfo> GlobalMapper::buildStripsInfo() const
{
  std::vector<ExtendedMappingInfo> out;
  for (int ide = 0; ide < o2::mid::detparams::NDetectionElements; ++ide) {
    for (int icol = mMapping.getFirstColumn(ide); icol < 7; ++icol) {
      int firstLine = mMapping.getFirstBoardBP(icol, ide);
      int nStrips = mMapping.getNStripsNBP(icol, ide);
      // NBP
      for (int istrip = 0; istrip < nStrips; ++istrip) {
        auto info = buildExtendedInfo(ide, icol, firstLine, istrip, 1);
        out.emplace_back(info);
      }
      // BP
      int lastLine = mMapping.getLastBoardBP(icol, ide);
      for (int iline = firstLine; iline <= lastLine; ++iline) {
        for (int istrip = 0; istrip < 16; ++istrip) {
          auto info = buildExtendedInfo(ide, icol, iline, istrip, 0);
          out.emplace_back(info);
        }
      }
    }
  }
  return out;
}

std::map<int, std::vector<std::pair<int, int>>> GlobalMapper::buildDEGeom() const
{
  std::map<int, std::vector<std::pair<int, int>>> out;
  std::pair<int, int> tmp;
  for (int ide = 0; ide < o2::mid::detparams::NDetectionElements; ++ide) {
    int icol = mMapping.getFirstColumn(ide);
    int iline = mMapping.getFirstBoardBP(icol, ide);
    bool isCutLow = (iline != 0);
    // (x1,y1)
    auto info = buildExtendedInfo(ide, icol, iline, 0, 0);
    out[ide].emplace_back(info.xpos, info.ypos);
    if (isCutLow) {
      tmp = {info.xpos + info.xwidth, info.ypos};
    }
    // (x1, y2)
    iline = mMapping.getLastBoardBP(icol, ide);
    info = buildExtendedInfo(ide, icol, iline, 15, 0);
    out[ide].emplace_back(info.xpos, info.ypos + info.ywidth);
    if (iline == 2) {
      // Cut high
      // (x1',y2)
      out[ide].emplace_back(info.xpos + info.xwidth, info.ypos + info.ywidth);
      info = buildExtendedInfo(ide, 1, 3, 15, 0);
      // (x1',y2')
      out[ide].emplace_back(info.xpos, info.ypos + info.ywidth);
    }
    // (x2,y2)
    info = buildExtendedInfo(ide, 6, 0, 15, 0);
    out[ide].emplace_back(info.xpos + info.xwidth, info.ypos + info.ywidth);
    // (x2,y1)
    info = buildExtendedInfo(ide, 6, 0, 0, 0);
    out[ide].emplace_back(info.xpos + info.xwidth, info.ypos);
    if (isCutLow) {
      // Cut low
      // (x1',y1')
      info = buildExtendedInfo(ide, 1, 0, 0, 0);
      out[ide].emplace_back(info.xpos, info.ypos);
      // (x1',y1)
      out[ide].emplace_back(tmp);
    }
    // (x1, y1)
    out[ide].emplace_back(out[ide].front());
  }
  return out;
}

} // namespace mid
} // namespace o2