// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.h
/// \brief Definition of the PHOS cluster finder
#ifndef ALICEO2_PHOS_CLUSTERER_H
#define ALICEO2_PHOS_CLUSTERER_H
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsPHOS/Cluster.h"
#include "PHOSReconstruction/FullCluster.h"
#include "PHOSCalib/CalibParams.h"
#include "PHOSCalib/BadChannelMap.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsPHOS/TriggerRecord.h"

namespace o2
{
namespace phos
{
class Geometry;

class Clusterer
{
 public:
  Clusterer() = default;
  ~Clusterer() = default;

  void initialize();
  void process(gsl::span<const Digit> digits, gsl::span<const TriggerRecord> dtr,
               const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
               std::vector<Cluster>* clusters, std::vector<TriggerRecord>* rigRec,
               o2::dataformats::MCTruthContainer<MCLabel>* cluMC);
  void processCells(gsl::span<const Cell> digits, gsl::span<const TriggerRecord> dtr,
                    const o2::dataformats::MCTruthContainer<MCLabel>* dmc, gsl::span<const uint> mcmap,
                    std::vector<Cluster>* clusters, std::vector<TriggerRecord>* rigRec,
                    o2::dataformats::MCTruthContainer<MCLabel>* cluMC);

  void makeClusters(gsl::span<const Digit> digits);
  void evalCluProperties(gsl::span<const Digit> digits, std::vector<Cluster>* clusters,
                         const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                         o2::dataformats::MCTruthContainer<MCLabel>* cluMC);

  float showerShape(float dx, float dz); // Parameterization of EM shower

  void makeUnfoldings(gsl::span<const Digit> digits); // Find and unfold clusters with few local maxima
  void unfoldOneCluster(FullCluster& iniClu, char nMax, gsl::span<int> digitId, gsl::span<const Digit> digits);

 protected:
  void convertCellsToDigits(gsl::span<const Cell> cells, int firstCellInEvent, int lastCellInEvent, gsl::span<const uint> mcmap);

  //Calibrate energy
  inline float calibrate(float amp, short absId) { return amp * mCalibParams->getGain(absId); }
  //Calibrate time
  inline float calibrateT(float time, short absId, bool isHighGain)
  {
    //Calibrate time
    if (isHighGain) {
      return time - mCalibParams->getHGTimeCalib(absId);
    } else {
      return time - mCalibParams->getLGTimeCalib(absId);
    }
  }
  //Test Bad map
  inline bool isBadChannel(short absId) { return (!mBadMap->isChannelGood(absId)); }

 protected:
  Geometry* mPHOSGeom = nullptr;             ///< PHOS geometry
  const CalibParams* mCalibParams = nullptr; //! Calibration coefficients
  const BadChannelMap* mBadMap = nullptr;    //! Calibration coefficients

  std::vector<FullCluster> mClusters; ///< internal vector of clusters
  int mFirstDigitInEvent;             ///< Range of digits from one event
  int mLastDigitInEvent;              ///< Range of digits from one event
  std::vector<Digit> mDigits;         ///< vector of trancient digits for cell processing
};
} // namespace phos
} // namespace o2

#endif
