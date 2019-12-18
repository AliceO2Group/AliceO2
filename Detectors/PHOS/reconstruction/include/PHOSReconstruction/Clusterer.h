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
  void process(const std::vector<Digit>* digits, const std::vector<TriggerRecord>* dtr,
               const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
               std::vector<Cluster>* clusters, std::vector<TriggerRecord>* rigRec,
               o2::dataformats::MCTruthContainer<MCLabel>* cluMC);
  void makeClusters(const std::vector<Digit>* digits);
  void evalCluProperties(const std::vector<Digit>* digits, std::vector<Cluster>* clusters,
                         const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                         o2::dataformats::MCTruthContainer<MCLabel>* cluMC);

  double showerShape(double dx, double dz); // Parameterization of EM shower

  void makeUnfoldings(const std::vector<Digit>* digits); // Find and unfold clusters with few local maxima
  void unfoldOneCluster(FullCluster* iniClu, int nMax, int* digitId, float* maxAtEnergy, const std::vector<Digit>* digits);

 protected:
  //Calibrate energy
  float calibrate(float amp, int absId);
  //Calibrate time
  float calibrateT(float time, int absId, bool isHighGain);
  //Test Bad map
  bool isBadChannel(int absId);

 protected:
  Geometry* mPHOSGeom = nullptr;             ///< PHOS geometry
  const CalibParams* mCalibParams = nullptr; //! Calibration coefficients
  const BadChannelMap* mBadMap = nullptr;    //! Calibration coefficients

  std::vector<FullCluster> mClusters; ///< internal vector of clusters
  int mFirstDigitInEvent;             ///< Range of digits from one event
  int mLastDigitInEvent;              ///< Range of digits from one event
};
} // namespace phos
} // namespace o2

#endif
