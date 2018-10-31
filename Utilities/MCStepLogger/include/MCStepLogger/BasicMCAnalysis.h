// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/* This class analyses properties of a simulation run which are directly accessible in 
 * the ROOT files created with the MCStepLogger. No complex parameters are derived from 
 * what is available and it therefore provides a first immediate conclusion on the 
 * simulation in terms of:
 *
 * -> number of simulated events
 * -> number of tracks
 *    -> total
 *    -> per PDG ID
 * -> number of steps (smallest accessible granularity of the particle transport)
 *    -> total
 *    -> per volume (absolute and relative)
 *    -> per PDG ID (absolute and relative)
 * -> spatial distribution of steps
 *    -> position
 *    -> steps in r-z plane
 *    -> step sizes
 *       -> all
 *       -> mean step size per event
 *       -> mean step size per volume per event
 *       -> mean step size per PDG ID per event
 * -> number of secondaries
 *    -> total
 *    -> per volume
 * -> number of calls to magnetic field
 * -> number of volumes traversed
 *
 * Note that all histograms are normalized to the number of simulated events in 
 * order to enable for an analysis comparison given 2 runs with different number 
 * of events (of course, statistics need to be large enough to make sure the phase
 * space is sufficiently scanned in both analyses in order to draw reliable 
 * conclusions from a comparison).
 * 
 * Nevertheless, this analysis can be used to compare simulations with benchmarked
 * events directly.
 */

#ifndef BASIC_MCANALYSIS_H_
#define BASIC_MCANALYSIS_H_

#include <unordered_map>

#include "MCStepLogger/MCAnalysis.h"

namespace o2
{
namespace mcstepanalysis
{

class BasicMCAnalysis : public MCAnalysis
{
 public:
  BasicMCAnalysis();

 protected:
  /// custom initialization of histograms
  void initialize() override;
  /// custom event loop
  void analyze(const std::vector<StepInfo>* const steps, const std::vector<MagCallInfo>* const magCalls) override;
  /// custom finalizations of produced histograms
  void finalize() override;

 private:
  // number of events
  TH1D* histNEvents;
  // number of tracks over all events
  TH1D* histNTracks;
  // number of tracks averaged over number of events
  TH1D* histNTracksPerEvent;
  // number of tracks mapped to particles' PDG ID averaged over number of events
  TH1D* histNTracksPerPDGPerEvent;
  // relative number of tracks mapped to particles' PDG ID averaged over number of events
  TH1D* histRelNTracksPerPDGPerEvent;
  // count total number of steps including all events
  TH1D* histNSteps;
  // number of steps averaged over number of events
  TH1D* histNStepsPerEvent;
  // number of steps per volume averaged over number of events
  TH1D* histNStepsPerVolPerEvent;
  // relative number of steps per volume averaged over number of events
  TH1D* histRelNStepsPerVolPerEvent;
  // number of steps made by particles of certain PDG ID averaged over number of events
  TH1D* histNStepsPerPDGPerEvent;
  // relative number of steps made by particles of certain PDG ID averaged over number of events
  TH1D* histRelNStepsPerPDGPerEvent;
  // histogram of x coordinates of steps averaged over number of events
  TH1D* histStepsXPerEvent;
  // histogram of y coordinates of steps averaged over number of events
  TH1D* histStepsYPerEvent;
  // histogram of z coordinates of steps averaged over number of events
  TH1D* histStepsZPerEvent;
  // average step size per event per volume averaged over number of events
  TH1D* histMeanStepSizePerVolPerEvent;
  // average step size per event per PDG averaged over number of events
  TH1D* histMeanStepSizePerPDGPerEvent;
  // overall average step size per event averaged over number of events
  TH1D* histMeanStepSizePerEvent;
  // counting all step sizes per event averaged over number of events
  TH1D* histStepSizesPerEvent;
  // steps in the r-z plane
  TH2D* histRZ;
  // total number of secondaries averaged over number of events
  TH1D* histNSecondariesPerEvent;
  // total number of secondaries per volume averaged over number of events
  TH1D* histNSecondariesPerVolPerEvent;
  // number of calls to magnetic field per volume averaged over number of events
  TH1D* histMagFieldCallsPerVolPerEvent;
  // number of calls to magnetic field (with small abs. value) per volume averaged over number of events
  TH1D* histSmallMagFieldCallsPerVolPerEvent;
  // count the number of volumes traversed
  TH1I* histNVols;
  // helper to keep track of all different volume IDs accross events
  std::vector<int> volIds;
  // helper to check in how many events a certain PDG was present
  std::unordered_map<std::string, float> pdgPresent;
  // helper to check in how many events a certain volume was traversed
  std::unordered_map<std::string, float> volPresent;

  ClassDefNV(MCAnalysis, 1);
};
} // namespace mcstepanalysis
} // namespace o2
#endif /* BASIC_MCANALYSIS_H_ */