// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/Digitizer.cxx
/// \brief  Implementation of the digitizer for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2018
#include "MIDSimulation/Digitizer.h"

namespace o2
{
namespace mid
{
//______________________________________________________________________________
Digitizer::Digitizer(const ChamberResponse& chamberResponse, const ChamberEfficiencyResponse& efficiencyResponse, const GeometryTransformer& transformer) : mGenerator(std::default_random_engine()), mRandom(), mResponse(chamberResponse), mEfficiencyResponse(efficiencyResponse), mMapping(), mTransformer(transformer), mDigits()
{
  /// Default constructor
}

//______________________________________________________________________________
void Digitizer::addStrip(const Mapping::MpStripIndex& stripIndex, int cathode, int deId)
{
  /// Adds a single digit
  for (auto& col : mDigits) {
    if (col.deId == deId && col.columnId == stripIndex.column) {
      col.addStrip(stripIndex.strip, cathode, stripIndex.line);
      return;
    }
  }

  mDigits.emplace_back(ColumnDataMC{(uint8_t)deId, (uint8_t)stripIndex.column});
  mDigits.back().setTimeStamp(mTime);
  mDigits.back().addStrip(stripIndex.strip, cathode, stripIndex.line);
}

//______________________________________________________________________________
bool Digitizer::addBPStrips(double xPos, double yPos, int deId, double prob, double xOffset)
{
  /// Checks the BP strips in the neighbour columns

  // The offset is used to check the strips in the bending plane
  // placed in the board close to the fired one
  // If the response says that the strip is too far away, there is no need to check further
  if (!mResponse.isFired(prob, std::abs(xOffset), 0, deId)) {
    return false;
  }

  Mapping::MpStripIndex stripIndex = mMapping.stripByPosition(xPos + 1.01 * xOffset, yPos, 0, deId, false);
  // Check if the point is still inside the RPC
  if (!stripIndex.isValid()) {
    return false;
  }
  addStrip(stripIndex, 0, deId);
  MpArea area = mMapping.stripByLocation(stripIndex.strip, 0, stripIndex.line, stripIndex.column, deId);
  std::array<double, 2> dist = {area.getYmax() - yPos, yPos - area.getYmin()};
  addNeighbours(stripIndex, 0, deId, prob, dist, xOffset);
  return true;
}

//______________________________________________________________________________
bool Digitizer::addNeighbours(const Mapping::MpStripIndex& stripIndex, int cathode, int deId, double prob,
                              const std::array<double, 2>& initialDist, double xOffset)
{
  /// Add neighbour strips

  double xOffset2 = xOffset * xOffset;
  for (int idir = 0; idir < 2; ++idir) {
    // Search for neighbours in the two directions
    // up and down for the BP, right and left for the NBP
    double dist = initialDist[idir];
    Mapping::MpStripIndex neigh = mMapping.nextStrip(stripIndex, cathode, deId, idir);
    while (neigh.isValid() && mResponse.isFired(prob, std::sqrt(dist * dist + xOffset2), cathode, deId)) {
      addStrip(neigh, cathode, deId);
      dist += mMapping.getStripSize(neigh.strip, cathode, neigh.column, deId);
      neigh = mMapping.nextStrip(neigh, cathode, deId, idir);
    }
  }
  return true;
}

//______________________________________________________________________________
bool Digitizer::hitToDigits(const Hit& hit)
{
  /// Generate digits from the hit

  // Clear
  mDigits.clear();

  // Convert point from global to local coordinates
  auto midPt = hit.middlePoint();
  int deId = hit.GetDetectorID();
  Point3D<double> localPoint = mTransformer.globalToLocal(deId, (double)midPt.x(), (double)midPt.y(), (double)midPt.z());

  // First get the touched BP strip
  Mapping::MpStripIndex stripIndex = mMapping.stripByPosition(localPoint.x(), localPoint.y(), 0, deId);

  // Check if the chamber was efficient
  bool isEfficientBP, isEfficientNBP;
  if (!mEfficiencyResponse.isEfficient(deId, stripIndex.column, stripIndex.line, isEfficientBP, isEfficientNBP)) {
    return false;
  }

  // Digitize if the RPC was efficient
  double prob = mRandom(mGenerator);

  if (isEfficientBP) {
    addStrip(stripIndex, 0, deId);
    MpArea area = mMapping.stripByLocation(stripIndex.strip, 0, stripIndex.line, stripIndex.column, deId);
    // This is the distance between the hit point and the edges of the strip along y
    std::array<double, 2> dist = {area.getYmax() - localPoint.y(), localPoint.y() - area.getYmin()};
    addNeighbours(stripIndex, 0, deId, prob, dist);
    // Search for neighbours in the close column toward inside
    addBPStrips(localPoint.x(), localPoint.y(), deId, prob, area.getXmin() - localPoint.x());
    // Search for neighbours in the close column toward outside
    addBPStrips(localPoint.x(), localPoint.y(), deId, prob, area.getXmax() - localPoint.x());
  }

  // Then check the touched NBP strip
  if (isEfficientNBP) {
    stripIndex = mMapping.stripByPosition(localPoint.x(), localPoint.y(), 1, deId);
    addStrip(stripIndex, 1, deId);
    MpArea area = mMapping.stripByLocation(stripIndex.strip, 1, stripIndex.line, stripIndex.column, deId);
    // This is the distance between the hit point and the edges of the strip along x
    std::array<double, 2> dist = {area.getXmax() - localPoint.x(), localPoint.x() - area.getXmin()};
    addNeighbours(stripIndex, 1, deId, prob, dist);
  }

  return true;
}

//______________________________________________________________________________
void Digitizer::process(const std::vector<Hit>& hits, std::vector<ColumnDataMC>& digitStore, o2::dataformats::MCTruthContainer<MCLabel>& mcContainer)
{
  /// Generate digits from a vector of hits
  digitStore.clear();
  mcContainer.clear();
  int firstStrip = 0, lastStrip = 0;

  for (auto& hit : hits) {
    hitToDigits(hit);

    for (auto& digit : mDigits) {
      digitStore.emplace_back(digit);
      for (int icathode = 0; icathode < 2; ++icathode) {
        if (getLabelLimits(icathode, digit, firstStrip, lastStrip)) {
          MCLabel label(hit.GetTrackID(), mEventID, mSrcID, digit.deId, digit.columnId, icathode, firstStrip, lastStrip);
          mcContainer.addElement(digitStore.size() - 1, label);
        }
      }
    }
  }
}

//______________________________________________________________________________
bool Digitizer::getLabelLimits(int cathode, const ColumnDataMC& col, int& firstStrip, int& lastStrip) const
{
  /// Gets the label limits
  int nLines = (cathode == 0) ? 4 : 1;
  int invalid = 9999;
  firstStrip = invalid, lastStrip = invalid;
  for (int iline = 0; iline < nLines; ++iline) {
    for (int istrip = 0; istrip < 16; ++istrip) {
      if (col.isStripFired(istrip, cathode, iline)) {
        lastStrip = MCLabel::getStrip(istrip, iline);
        if (firstStrip == invalid) {
          firstStrip = lastStrip;
        }
      } else if (firstStrip != invalid) {
        return true;
      }
    }
  }
  return (firstStrip != invalid);
}

//______________________________________________________________________________
Digitizer createDefaultDigitizer()
{
  return Digitizer(createDefaultChamberResponse(), createDefaultChamberEfficiencyResponse(), createDefaultTransformer());
}

} // namespace mid
} // namespace o2
