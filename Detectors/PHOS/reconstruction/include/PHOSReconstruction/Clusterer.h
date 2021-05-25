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
#include "DataFormatsPHOS/CalibParams.h"
#include "DataFormatsPHOS/BadChannelsMap.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"

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
               std::vector<Cluster>& clusters, std::vector<CluElement>& cluel, std::vector<TriggerRecord>& rigRec,
               o2::dataformats::MCTruthContainer<MCLabel>& cluMC);
  void processCells(gsl::span<const Cell> digits, gsl::span<const TriggerRecord> dtr,
                    const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                    std::vector<Cluster>& clusters, std::vector<CluElement>& cluel, std::vector<TriggerRecord>& rigRec,
                    o2::dataformats::MCTruthContainer<MCLabel>& cluMC);

  void makeClusters(std::vector<Cluster>& clusters, std::vector<o2::phos::CluElement>& cluel);

  void setBadMap(std::unique_ptr<BadChannelsMap>& m) { mBadMap = std::move(m); }
  void setCalibration(std::unique_ptr<CalibParams>& c) { mCalibParams = std::move(c); }

 protected:
  //Calibrate energy
  inline float calibrate(float amp, short absId, bool isHighGain)
  {
    if (isHighGain) {
      return amp * mCalibParams->getGain(absId);
    } else {
      return amp * mCalibParams->getGain(absId) * mCalibParams->getHGLGRatio(absId);
    }
  }
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

  char getNumberOfLocalMax(Cluster& clu, std::vector<CluElement>& cluel);
  void evalAll(Cluster& clu, std::vector<CluElement>& cluel) const;
  void evalLabels(std::vector<Cluster>& clusters, std::vector<CluElement>& cluel,
                  const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                  o2::dataformats::MCTruthContainer<MCLabel>& cluMC);

  double showerShape(double r2, double& deriv); // Parameterization of EM shower

  void makeUnfolding(Cluster& clu, std::vector<Cluster>& clusters, std::vector<o2::phos::CluElement>& cluel); //unfold cluster with few local maxima
  void unfoldOneCluster(Cluster& iniClu, char nMax, std::vector<Cluster>& clusters, std::vector<CluElement>& cluelements);

 protected:
  static constexpr short NLOCMAX = 30; //Maximal number of local maxima in cluster
  bool mProcessMC = false;
  int miCellLabel = 0;
  bool mFullCluOutput = false;               ///< Write output full of reduced (no contributed digits) clusters
  Geometry* mPHOSGeom = nullptr;             ///< PHOS geometry
  std::unique_ptr<CalibParams> mCalibParams; ///! Calibration coefficients
  std::unique_ptr<BadChannelsMap> mBadMap;   ///! Bad map

  std::vector<CluElement> mCluEl; ///< internal vector of clusters
  std::vector<Digit> mTrigger;    ///< internal vector of clusters
  int mFirstElememtInEvent;       ///< Range of digits from one event
  int mLastElementInEvent;        ///< Range of digits from one event

  std::vector<float> mProp;             ///< proportion of clusters in the current digit
  std::array<float, NLOCMAX> mxMax;     ///< current maximum coordinate
  std::array<float, NLOCMAX> mzMax;     ///< in the unfolding procedure
  std::array<float, NLOCMAX> meMax;     ///< currecnt amplitude in unfoding
  std::array<float, NLOCMAX> mxMaxPrev; ///< coordunates at previous step
  std::array<float, NLOCMAX> mzMaxPrev; ///< coordunates at previous step
  std::array<float, NLOCMAX> mdx;       ///< step on current minimization iteration
  std::array<float, NLOCMAX> mdz;       ///< step on current minimization iteration
  std::array<float, NLOCMAX> mdxprev;   ///< step on previoud minimization iteration
  std::array<float, NLOCMAX> mdzprev;   ///< step on previoud minimization iteration
  std::array<double, NLOCMAX> mA;       ///< transient variable for derivative calculation
  std::array<double, NLOCMAX> mxB;      ///< transient variable for derivative calculation
  std::array<double, NLOCMAX> mzB;      ///< transient variable for derivative calculation
  std::array<double, NLOCMAX> mfijx;    ///< transient variable for derivative calculation
  std::array<double, NLOCMAX> mfijz;    ///< transient variable for derivative calculation
  std::array<double, NLOCMAX> mfijr;    ///< transient variable for derivative calculation
  std::array<double, NLOCMAX> mfij;     ///< transient variable for derivative calculation
  std::vector<bool> mIsLocalMax;        ///< transient array for local max finding
  std::array<int, NLOCMAX> mMaxAt;      ///< indexes of local maxima
};
} // namespace phos
} // namespace o2

#endif
