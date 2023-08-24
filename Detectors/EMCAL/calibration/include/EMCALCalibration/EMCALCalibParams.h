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

/// \class EMCALCalibInitParams
/// \brief  Init parameters for emcal calibrations
/// \author Joshua Koenig
/// \ingroup EMCALCalib
/// \since Apr 5, 2022

#ifndef EMCAL_CALIB_INIT_PARAMS_H_
#define EMCAL_CALIB_INIT_PARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace emcal
{

// class containing the parameters to trigger the calibrations
struct EMCALCalibParams : public o2::conf::ConfigurableParamHelper<EMCALCalibParams> {

  // parameters for bad channel calibration
  unsigned int minNEvents_bc = 1e7;     ///< minimum number of events to trigger the calibration
  unsigned int minNEntries_bc = 1e6;    ///< minimum number of entries to trigger the calibration
  bool useNEventsForCalib_bc = true;    ///< use the minimum number of events to trigger the calibration
  bool useScaledHisto_bc = true;        ///< use the scaled histogram for the bad channel map
  bool enableTestMode_bc = false;       ///< enable test mode for calibration
  int nBinsEnergyAxis_bc = 1000;        ///< number of bins for boost histogram energy axis
  bool useTimeInfoForCalib_bc = true;   ///< weather to use the timing information as a criterion in the bad channel analysis
  float maxValueEnergyAxis_bc = 40;     ///< maximum value for boost histogram energy axis (minimum is always 0)
  int nBinsTimeAxis_bc = 1000;          ///< number of bins for boost histogram time axis
  float rangeTimeAxisLow_bc = -500;     ///< minimum value of time for histogram range
  float rangeTimeAxisHigh_bc = 500;     ///< maximum value of time for histogram range
  float minCellEnergyTime_bc = 0.1;     ///< minimum energy needed to fill the time histogram
  float sigmaTime_bc = 5;               ///< sigma value for the upper cut on the time-variance distribution
  unsigned int slotLength_bc = 0;       ///< Lenght of the slot before calibration is triggered. If set to 0 calibration is triggered when hasEnoughData returns true
  bool UpdateAtEndOfRunOnly_bc = false; ///< switch to enable trigger of calibration only at end of run
  float minNHitsForMeanEnergyCut = 100; ///< mean number of hits per cell that is needed to cut on the mean energy per hit. Needed for high energy intervals as outliers can distort the distribution
  float minNHitsForNHitCut = 1;         ///< mean number of hits per cell that is needed to cut on the mean number of hits. Needed for high energy intervals as outliers can distort the distribution
  float minCellEnergy_bc = 0.1;         ///< minimum cell energy considered for filling the histograms for bad channel calib. Should speedup the filling of the histogram to suppress noise
  float fractionEvents_bc = 1.;         ///< fraction of events used in bad channel calibration

  // parameters for time calibration
  unsigned int minNEvents_tc = 1e7;      ///< minimum number of events to trigger the calibration
  unsigned int minNEntries_tc = 1e6;     ///< minimum number of entries to trigger the calibration
  bool useNEventsForCalib_tc = true;     ///< use the minimum number of events to trigger the calibration
  float minCellEnergy_tc = 0.5;          ///< minimum cell energy to enter the time calibration (typical minimum seed energy for clusters), time resolution gets better with rising energy
  int minTimeForFit_tc = -300;           ///< minimum cell time considered for the time calibration in ns
  int maxTimeForFit_tc = 300;            ///< maximum cell time considered for the time calibration in ns
  int restrictFitRangeToMax_tc = 25;     ///< window around the largest entry within the minTimeForFit in which the fit is performed in ns
  int nBinsTimeAxis_tc = 1500;           ///< number of bins used for the time calibration
  int minValueTimeAxis_tc = -500;        ///< minimum value of the time axis in the time calibration
  int maxValueTimeAxis_tc = 1000;        ///< maximum value of the time axis in the time calibration
  float lowGainOffset_tc = 8.8;          ///< Offset between high gain and low gain in ns. This is needed since the low gain calib will not be possible due to insuficient statistics
  unsigned int slotLength_tc = 0;        ///< Lenght of the slot before calibration is triggered. If set to 0 calibration is triggered when hasEnoughData returns true
  bool UpdateAtEndOfRunOnly_tc = false;  ///< switch to enable trigger of calibration only at end of run
  float maxAllowedDeviationFromMax = 10; ///< maximum deviation allowed between the estimated maximum of the fit and the true maximum from the distribution. If deviation is larger then value, the fit likely failed. In this case, the true value is taken
  float fractionEvents_tc = 1.;          ///< fraction of events used in time calibration

  // common parameters
  std::string calibType = "time";                      ///< type of calibration to run
  std::string localRootFilePath = "";                  ///< path to local root file in order to store the calibration histograms (off by default, only to be used for testing)
  bool enableFastCalib = false;                        ///< switch to enable fast calibration. Instead of filling boost histograms, mean and sigma of cells is calculated on the fly
  bool enableTimeProfiling = false;                    ///< enable to log how much time is spent in the run function in the calibrator spec. Needed for speed tests offline and at point 2
  bool setSavedSlotAllowed_EMC = true;                 ///< if true, saving and loading of calibrations from last run and for next run is enabled
  bool setSavedSlotAllowedSOR_EMC = true;              ///< if true, stored calibrations from last run can be loaded in the next run (if false, storing of the calib histograms is still active in contrast to setSavedSlotAllowed_EMC)
  long endTimeMargin = 2592000000;                     ///< set end TS to 30 days after slot ends (1000 * 60 * 60 * 24 * 30)
  std::string selectedClassMasks = "C0TVX-B-NOPF-EMC"; ///< name of EMCal min. bias trigger that is used for calibration

  // old parameters. Keep them for a bit (can be deleted after september 5th) as otherwise ccdb and o2 version might not be in synch
  unsigned int minNEvents = 1e7;              ///< minimum number of events to trigger the calibration
  unsigned int minNEntries = 1e6;             ///< minimum number of entries to trigger the calibration
  bool useNEventsForCalib = true;             ///< use the minimum number of events to trigger the calibration
  bool useScaledHistoForBadChannelMap = true; ///< use the scaled histogram for the bad channel map
  bool enableTestMode = false;                ///< enable test mode for calibration
  float minCellEnergyForTimeCalib = 0.5;      ///< minimum cell energy to enter the time calibration (typical minimum seed energy for clusters), time resolution gets better with rising energy
  unsigned int slotLength = 0;                ///< Lenght of the slot before calibration is triggered. If set to 0 calibration is triggered when hasEnoughData returns true
  bool UpdateAtEndOfRunOnly = false;          ///< switsch to enable trigger of calibration only at end of run
  int minTimeForFit = -300;                   ///< minimum cell time considered for the time calibration in ns
  int maxTimeForFit = 300;                    ///< maximum cell time considered for the time calibration in ns
  int restrictFitRangeToMax = 25;             ///< window around the largest entry within the minTimeForFit in which the fit is performed in ns

  O2ParamDef(EMCALCalibParams, "EMCALCalibParams");
};

} // namespace emcal

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::emcal::EMCALCalibParams> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif /*EMCAL_CALIB_INIT_PARAMS_H_ */
