// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <algorithm>
#include <iostream>

#include "MCStepLogger/BasicMCAnalysis.h"
#include "MCStepLogger/MCAnalysisUtilities.h"

ClassImp(o2::mcstepanalysis::BasicMCAnalysis);

using namespace o2::mcstepanalysis;

BasicMCAnalysis::BasicMCAnalysis()
  : MCAnalysis("BasicMCAnalysis")
{
}

void BasicMCAnalysis::initialize()
{
  // number of events
  histNEvents = getHistogram<TH1D>("nEvents", 1, 0., 1.);
  // number of tracks over all events
  histNTracks = getHistogram<TH1D>("nTracks", 1, 0., 1.);
  // number of tracks averaged over number of events
  histNTracksPerEvent = getHistogram<TH1D>("nTracksPerEvent", 1, 0., 1.);
  // number of tracks mapped to particles' PDG ID averaged over number of events
  histNTracksPerPDGPerEvent = getHistogram<TH1D>("nTracksPerPDGPerEvent", 1, 0., 1.);
  // relative number of tracks mapped to particles' PDG ID averaged over number of events
  histRelNTracksPerPDGPerEvent = getHistogram<TH1D>("relNTracksPerPDGPerEvent", 1, 0., 1.);
  // count total number of steps including all events
  histNSteps = getHistogram<TH1D>("nSteps", 1, 0., 1.);
  // number of steps averaged over number of events
  histNStepsPerEvent = getHistogram<TH1D>("nStepsPerEvent", 1, 0., 1.);
  // number of steps per volume averaged over number of events
  histNStepsPerVolPerEvent = getHistogram<TH1D>("nStepsPerVolPerEvent", 1, 0., 1.);
  // relative number of steps per volume averaged over number of events
  histRelNStepsPerVolPerEvent = getHistogram<TH1D>("relNStepsPerVolPerEvent", 1, 0., 1.);
  // number of steps made by particles of certain PDG ID averaged over number of events
  histNStepsPerPDGPerEvent = getHistogram<TH1D>("nStepsPerPDGPerEvent", 1, 0., 1.);
  // relative number of steps made by particles of certain PDG ID averaged over number of events
  histRelNStepsPerPDGPerEvent = getHistogram<TH1D>("relNStepsPerPDGPerEvent", 1, 0., 1.);
  // histogram of x coordinates of steps averaged over number of events
  histStepsXPerEvent = getHistogram<TH1D>("stepsXPerEvent", 100, -2100., 2100.);
  // histogram of y coordinates of steps averaged over number of events
  histStepsYPerEvent = getHistogram<TH1D>("stepsYPerEvent", 100, -2100., 2100.);
  // histogram of z coordinates of steps averaged over number of events
  histStepsZPerEvent = getHistogram<TH1D>("stepsZPerEvent", 100, -3100., 3100.);
  // overall average step size per event averaged over number of events
  histMeanStepSizePerEvent = getHistogram<TH1D>("meanStepSizePerEvent", 1, 0., 1.);
  // average step size per event per volume averaged over number of events
  histMeanStepSizePerVolPerEvent = getHistogram<TH1D>("meanStepSizePerVolPerEvent", 1, 0., 1.);
  // average step size per event per PDG averaged over number of events
  histMeanStepSizePerPDGPerEvent = getHistogram<TH1D>("meanStepSizePerPDGPerEvent", 1, 0., 1.);
  // counting all step sizes per event averaged over number of events
  histStepSizesPerEvent = getHistogram<TH1D>("stepSizesPerEvent", 50, 0., 250.);
  // steps in the r-z plane
  histRZ = getHistogram<TH2D>("RZOccupancy", 200, -3000., 3000., 200, 0., 3000.);
  // total number of secondaries averaged over number of events
  histNSecondariesPerEvent = getHistogram<TH1D>("nSecondariesPerEvent", 1, 0., 1.);
  // number of secondaries per volume averaged over number of events
  histNSecondariesPerVolPerEvent = getHistogram<TH1D>("nSecondariesPerVolPerEvent", 1, 0., 1.);
  // number of calls to magnetic field per volume averaged over number of events
  histMagFieldCallsPerVolPerEvent = getHistogram<TH1D>("magFieldCallsPerVolPerEvent", 1, 0., 1.);
  // number of calls to magnetic field (with small abs. value) per volume averaged over number of events
  histSmallMagFieldCallsPerVolPerEvent = getHistogram<TH1D>("smallMagFieldCallsPerVolPerEvent", 1, 0., 1.);
  // count the number of volumes traversed
  histNVols = getHistogram<TH1I>("nVolumes", 1, 0., 1.);
  // helper to keep track of all different volume IDs accross events
  volIds.clear();
  // helper to check in how many events a certain PDG was present
  volPresent.clear();
  // helper to check in how many events a certain volume was traversed
  pdgPresent.clear();
}

void BasicMCAnalysis::analyze(const std::vector<StepInfo>* const steps, const std::vector<MagCallInfo>* const magCalls)
{
  // first of all, count events
  histNEvents->Fill(0.5);
  // to store the volume name
  std::string volName = "";
  // loop over magnetic field calls
  for (const auto& call : *magCalls) {
    if (call.stepid < 0) {
      continue;
    }
    auto step = steps->operator[](call.stepid);
    mAnalysisManager->getLookupVolName(step.volId, volName);
    if (call.B < 0.01) {
      histSmallMagFieldCallsPerVolPerEvent->Fill(volName.c_str(), 1.);
    }
    histMagFieldCallsPerVolPerEvent->Fill(volName.c_str(), 1.);
  }

  // total number of steps in this event
  int nSteps = 0;
  // to store particle ID
  int pdgId = 0;
  // get the mean step size per event
  float meanStepSizePerEvent = 0.;
  // monitor number of different tracks
  std::vector<int> tracks;
  // step sizes per volume id in one event
  std::vector<float> stepSizesPerVol;
  // number of steps per volume ID in this event
  std::vector<int> nStepsPerVol;
  // steps sizes per PDG ID in one event, use a map here since PDG IDs can be large (~10^10)
  // don't attempt to handle such a vector
  std::unordered_map<int, float> stepSizesPerPDGMap;
  // number of steps per PDG ID in this event
  std::unordered_map<int, int> nStepsPerPDGMap;

  // loop over all steps in an event
  for (const auto& step : *steps) {
    // prepare for PDG ids and volume names
    mAnalysisManager->getLookupPDG(step.trackID, pdgId);
    mAnalysisManager->getLookupVolName(step.volId, volName);
    // first increment the total number of steps over all events
    nSteps++;

    // avoid double counting of tracks in an event, so check if track ID is already registered
    if (std::find(tracks.begin(), tracks.end(), step.trackID) == tracks.end() && step.trackID > -1) {
      // if not, there is a new track
      tracks.push_back(step.trackID);
      // increment track per PDG ID and treat PDGs as alphanumeric labels, so we can easily deflate later
      histNTracksPerPDGPerEvent->Fill(std::to_string(pdgId).c_str(), 1.);
    }
    // summarize all volume Ids over all events to see, how many volumes were traversed in total
    if (std::find(volIds.begin(), volIds.end(), step.volId) == volIds.end() && step.volId > -1) {
      volIds.push_back(step.volId);
    }

    // get the spatial coordinates of this step
    histStepsXPerEvent->Fill(step.x);
    histStepsYPerEvent->Fill(step.y);
    histStepsZPerEvent->Fill(step.z);
    histRZ->Fill(step.z, std::sqrt(step.x * step.x + step.y * step.y));
    // extract the step length
    float stepLength = 0.;
    // be save and check whether there are e.g. NaNs (happened!)
    if (std::isfinite(step.step) && step.step > 0.) {
      stepLength = step.step;
    }
    // summarise all steps sizes
    histStepSizesPerEvent->Fill(stepLength);
    // summarise steps sizes per volume in this event
    if (stepSizesPerVol.size() <= step.volId) {
      stepSizesPerVol.resize(step.volId + 1, 0.);
      nStepsPerVol.resize(step.volId + 1, 0);
    }
    stepSizesPerVol[step.volId] += stepLength;
    nStepsPerVol[step.volId]++;
    // summarise step sizes per PDG ID in this event
    if (stepSizesPerPDGMap.find(pdgId) == stepSizesPerPDGMap.end()) {
      stepSizesPerPDGMap[pdgId] = stepLength;
      nStepsPerPDGMap[pdgId] = 1;
    } else {
      stepSizesPerPDGMap[pdgId] += stepLength;
      nStepsPerPDGMap[pdgId]++;
    }

    // secondaries
    histNSecondariesPerEvent->Fill(0.5, step.nsecondaries);
    histNSecondariesPerVolPerEvent->Fill(volName.c_str(), step.nsecondaries);
  }
  // add number of steps
  histNSteps->Fill(0.5, nSteps);
  histNSteps->SetEntries(nSteps);
  histNStepsPerEvent->Fill(0.5, nSteps);
  histNStepsPerEvent->SetEntries(nSteps);

  // add number of tracks
  histNTracks->Fill(0.5, tracks.size());
  histNTracks->SetEntries(tracks.size());
  histNTracksPerEvent->Fill(0.5, tracks.size());
  histNTracksPerEvent->SetEntries(tracks.size());
  // update number of steps, number of steps per volume, mean step length and mean step length per volume
  float meanStepSizes = 0.;
  for (int i = 0; i < nStepsPerVol.size(); i++) {
    // if no step was made with that volume id, continue
    if (nStepsPerVol[i] < 1) {
      continue;
    }
    mAnalysisManager->getLookupVolName(i, volName);
    // number of steps per volume
    histNStepsPerVolPerEvent->Fill(volName.c_str(), nStepsPerVol[i]);
    meanStepSizes += stepSizesPerVol[i];
    histMeanStepSizePerVolPerEvent->Fill(volName.c_str(), stepSizesPerVol[i] / float(nStepsPerVol[i]));
    // check in how many events a certain volume was present
    if (volPresent.find(volName) == volPresent.end()) {
      volPresent.insert(std::pair<std::string, float>(volName, 1.));
    } else {
      volPresent[volName]++;
    }
  }
  histNStepsPerVolPerEvent->SetEntries(nSteps);
  histMeanStepSizePerEvent->Fill(0.5, meanStepSizes / float(nSteps));
  // since the step size is the difference between 2 points in 3D space,
  // the exact number of entries is nSteps-nTracks, however nTracks << nSteps is expected
  histMeanStepSizePerEvent->SetEntries(nSteps);
  histMeanStepSizePerVolPerEvent->SetEntries(nSteps);
  // number of steps per PDG ID and mean step length per PDG ID
  for (auto& sp : nStepsPerPDGMap) {
    std::string pdgString(std::to_string(sp.first));
    // number of steps per volume
    histNStepsPerPDGPerEvent->Fill(pdgString.c_str(), sp.second);
    histMeanStepSizePerPDGPerEvent->Fill(pdgString.c_str(), stepSizesPerPDGMap[sp.first] / float(sp.second));
    // check in how many events a certain volume was present
    if (pdgPresent.find(pdgString) == pdgPresent.end()) {
      pdgPresent.insert(std::pair<std::string, float>(pdgString, 1.));
    } else {
      pdgPresent[pdgString]++;
    }
  }
  histNStepsPerPDGPerEvent->SetEntries(nSteps);
  histMeanStepSizePerPDGPerEvent->SetEntries(nSteps);
}

void BasicMCAnalysis::finalize()
{
  // fill and update (e.g. scaling) some histograms
  histNVols->Fill(0.5, volIds.size());
  histNVols->SetEntries(volIds.size());

  // just normalise over all events
  histNStepsPerEvent->Scale(1. / histNEvents->GetEntries());
  histNTracksPerEvent->Scale(1. / histNEvents->GetEntries());
  histNSecondariesPerEvent->Scale(1. / histNEvents->GetEntries());
  histStepsXPerEvent->Scale(1. / histNEvents->GetEntries());
  histStepsYPerEvent->Scale(1. / histNEvents->GetEntries());
  histStepsZPerEvent->Scale(1. / histNEvents->GetEntries());
  histStepSizesPerEvent->Scale(1. / histNEvents->GetEntries());
  histMeanStepSizePerEvent->Scale(1. / histNEvents->GetEntries());
  // normalize per volume to number of events where a certain volume was present
  for (auto& vp : volPresent) {
    vp.second = 1. / vp.second;
  }
  for (auto& pp : pdgPresent) {
    pp.second = 1. / pp.second;
  }
  // scale histograms bin-wise
  utilities::scalePerBin(histNStepsPerVolPerEvent, volPresent);
  utilities::scalePerBin(histMagFieldCallsPerVolPerEvent, volPresent);
  utilities::scalePerBin(histSmallMagFieldCallsPerVolPerEvent, volPresent);
  utilities::scalePerBin(histNSecondariesPerVolPerEvent, volPresent);
  utilities::scalePerBin(histMeanStepSizePerVolPerEvent, volPresent);
  utilities::scalePerBin(histMeanStepSizePerPDGPerEvent, pdgPresent);
  utilities::scalePerBin(histNStepsPerPDGPerEvent, pdgPresent);
  utilities::scalePerBin(histNTracksPerPDGPerEvent, pdgPresent);

  int nVolBins = histNStepsPerVolPerEvent->GetNbinsX();
  for (int i = 1; i < nVolBins + 1; i++) {
    histRelNStepsPerVolPerEvent->Fill(histNStepsPerVolPerEvent->GetXaxis()->GetBinLabel(i), histNStepsPerVolPerEvent->GetBinContent(i));
  }
  histRelNStepsPerVolPerEvent->Scale(1. / double(histNStepsPerEvent->GetMaximum()));
  histRelNStepsPerVolPerEvent->SetEntries(histNSteps->GetEntries());

  int nPDGBins = histNTracksPerPDGPerEvent->GetNbinsX();
  for (int i = 1; i < nPDGBins + 1; i++) {
    histRelNTracksPerPDGPerEvent->Fill(histNTracksPerPDGPerEvent->GetXaxis()->GetBinLabel(i), histNTracksPerPDGPerEvent->GetBinContent(i));
  }
  histRelNTracksPerPDGPerEvent->Scale(1. / double(histNTracksPerEvent->GetMaximum()));
  histRelNTracksPerPDGPerEvent->SetEntries(histNTracks->GetEntries());

  nPDGBins = histNStepsPerPDGPerEvent->GetNbinsX();
  for (int i = 1; i < nPDGBins + 1; i++) {
    histRelNStepsPerPDGPerEvent->Fill(histNStepsPerPDGPerEvent->GetXaxis()->GetBinLabel(i), histNStepsPerPDGPerEvent->GetBinContent(i));
  }
  histRelNStepsPerPDGPerEvent->Scale(1. / double(histNStepsPerEvent->GetMaximum()));
  histRelNStepsPerPDGPerEvent->SetEntries(histNSteps->GetEntries());
}
