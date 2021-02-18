// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/Digitizer.h
/// \brief  Digitizer for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2018
#ifndef O2_MID_DIGITIZER_H
#define O2_MID_DIGITIZER_H

#include <random>
#include <vector>
#include <array>
#include "MathUtils/Cartesian.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MIDBase/Mapping.h"
#include "MIDBase/GeometryTransformer.h"
#include "MIDSimulation/ChamberResponse.h"
#include "MIDSimulation/ChamberEfficiencyResponse.h"
#include "MIDSimulation/Hit.h"
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/MCLabel.h"

namespace o2
{
namespace mid
{
class Digitizer
{
 public:
  Digitizer(const ChamberResponse& chamberResponse, const ChamberEfficiencyResponse& efficiencyResponse, const GeometryTransformer& transformer);
  virtual ~Digitizer() = default;

  void process(const std::vector<Hit>& hits, std::vector<ColumnDataMC>& digitStore, o2::dataformats::MCTruthContainer<MCLabel>& mcContainer);

  /// Sets the event ID
  void setEventID(int entryID) { mEventID = entryID; }

  /// Sets the source ID
  void setSrcID(int sourceID) { mSrcID = sourceID; }

  /// Sets the geometry transformer
  void setGeometryTransformer(const GeometryTransformer& transformer) { mTransformer = transformer; }

  /// Sets the chamber response
  void setChamberResponse(const ChamberResponse& chamberResponse) { mResponse = chamberResponse; }

  /// Sets the seed
  void setSeed(unsigned int seed) { mGenerator.seed(seed); }

 private:
  void addStrip(const Mapping::MpStripIndex& stripIndex, int cathode, int deId);
  bool addBPStrips(double xPos, double yPos, int deId, double prob, double xOffset);
  bool addNeighbours(const Mapping::MpStripIndex& stripIndex, int cathode, int deId, double prob,
                     const std::array<double, 2>& initialDist, double xOffset = 0.);
  bool hitToDigits(const Hit& hit);
  bool getLabelLimits(int cathode, const ColumnDataMC& col, int& firstStrip, int& lastStrip) const;

  int mEventID{0};
  int mSrcID{0};

  std::default_random_engine mGenerator;          ///< Random numbers generator
  std::uniform_real_distribution<double> mRandom; ///< Uniform distribution
  ChamberResponse mResponse;                      ///< Chamber response
  ChamberEfficiencyResponse mEfficiencyResponse;  ///< Chamber efficiency response
  Mapping mMapping;                               ///< Mapping
  GeometryTransformer mTransformer;               ///< Geometry transformer
  std::vector<ColumnDataMC> mDigits;              /// Digits per hit
};

Digitizer createDefaultDigitizer();

} // namespace mid
} // namespace o2

#endif /* O2_MID_DIGITIZER_H */
