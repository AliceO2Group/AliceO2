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

#include "TMath.h"

#include "TRDWorkflow/TRDEventDisplayFeedSpec.h"

#include "DataFormatsTRD/TrackTriggerRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DataFormatsParameters/GRPObject.h"

#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include "DetectorsCommonDataFormats/DetID.h"

using namespace o2::framework;
using namespace o2::globaltracking;

namespace o2
{
namespace trd
{

void TRDEventDisplayFeedSpec::init(o2::framework::InitContext& ic)
{
  LOG(info) << "Initializing event display feed...";
}

json TRDEventDisplayFeedSpec::getTracksJson(gsl::span<const TrackTRD> tracks, gsl::span<const Tracklet64> tracklets, gsl::span<const TrackTriggerRecord> trackTrigRecs, int iEvent)
{
  json trackArray = json::array();

  const auto& trigRec = trackTrigRecs[iEvent];
  for (int iTrack = 0; iTrack < trigRec.getNumberOfTracks(); ++iTrack) {
    TrackTRD track = tracks[iTrack + trigRec.getFirstTrack()];
    std::string trackId = Form("E%d_T%d", iEvent, iTrack);

    float alpha = track.getAlpha();
    if (alpha < 0) {
      alpha += TMath::Pi() * 2;
    }

    int sector = TMath::Nint(18.0 * alpha / (2 * TMath::Pi()) - 0.5);
    int stack;
    for (int iLayer = 0; iLayer < 6; iLayer++) {
      int trackletIdx = track.getTrackletIndex(iLayer);
      if (trackletIdx != -1) {
        int detector = tracklets[trackletIdx].getDetector();
        stack = detector % 30 / 6;
        break;
      }
    }

    // Tangent of the track momentum dip angle
    float tanLambda = track.getParam(3);
    float lambdaDeg = TMath::ATan(tanLambda) * 180 / TMath::Pi();

    // Record stacks with tracks for look-up by writeDigits() later
    int layerZero = mGeo->getDetector(0, stack, sector);
    mUsedDetectors.set(layerZero);

    json trackJson = {
      {"id", trackId},
      {"stk", stack},
      {"sec", sector},
      {"typ", "Trd"},
      {"i", {{"pT", track.getPt()}, {"alpha", track.getAlpha()}, {"lambda", lambdaDeg}, {"pid", (int)track.getPID().getID()}}},
      {"path", json::array()},
      {"tlids", json::array()}};

    bool ok;
    for (int x = 1; x <= 470; x += 10) {
      auto xyz = track.getXYZGloAt(x, mBz, ok);
      json point = {{"x", xyz.X()}, {"y", xyz.Y()}, {"z", xyz.Z()}};
      trackJson["path"].push_back(point);
    }

    // Match TRD tracklets to track
    for (int iLayer = 0; iLayer < 6; iLayer++) {
      // trackletIdx gives absolute index of tracklet across all events
      int trackletIdx = track.getTrackletIndex(iLayer);
      if (trackletIdx != -1) {
        int trackletCount = 0;
        // count number of tracklets in all events prior to iEvent
        for (int i = 0; i < iEvent; i++) {
          trackletCount += mTrigRecs[i].getNumberOfTracklets();
        }
        // trackletId needs relative tracklet index within single event
        std::string trackletId = Form("E%d_L%d", iEvent, trackletIdx - trackletCount);
        // record {trackletId, trackId} pair in order to match tracks to tracklets in printTracklets()
        mTrackletMap.insert(std::pair<std::string, std::string>(trackletId, trackId));
        trackJson["tlids"].push_back(trackletId);
      }
    }
    trackArray.push_back(trackJson);
  }
  return trackArray;
}

json TRDEventDisplayFeedSpec::getTrackletsJson(gsl::span<const Tracklet64> tracklets, int iEvent)
{
  json trackletArray = json::array();

  const auto& trigRec = mTrigRecs[iEvent];
  for (int iTracklet = 0; iTracklet < trigRec.getNumberOfTracklets(); ++iTracklet) {
    Tracklet64 tracklet = tracklets[iTracklet + trigRec.getFirstTracklet()];
    CalibratedTracklet cTracklet = mTransformer.transformTracklet(tracklet);

    std::string trackletId = Form("E%d_L%d", iEvent, iTracklet);
    // Find matched track if it exists
    std::string trackId = (mTrackletMap.find(trackletId) != mTrackletMap.end() ? mTrackletMap.at(trackletId) : "null");

    int detector = tracklet.getDetector();
    int sector = mGeo->getSector(detector);
    int stack = mGeo->getStack(detector);
    int layer = mGeo->getLayer(detector);

    // Start position of both raw and calibrated tracklet in event display
    float localY = cTracklet.getY();
    // Slope of raw tracklet (key dyDxAN in JSON)
    float rawDyDx = tracklet.getUncalibratedDy() / mGeo->cdrHght();
    // Slope of calibrated tracklet
    float dyDx = cTracklet.getDy() / mGeo->cdrHght();

    json trackletJson = {
      {"id", trackletId},
      {"stk", stack},
      {"sec", sector},
      {"lyr", layer},
      {"row", tracklet.getPadRow()},
      {"trk", trackId},
      {"lY", localY},
      {"dyDx", dyDx},
      {"dyDxAN", rawDyDx}};

    if (trackId == "null") {
      trackletJson["trk"] = nullptr;
    }

    trackletArray.push_back(trackletJson);
  }
  return trackletArray;
}

void TRDEventDisplayFeedSpec::writeDigits(gsl::span<const Digit> digits, int iEvent)
{
  const auto& trigRec = mTrigRecs[iEvent];
  for (int det = 0; det < constants::MAXCHAMBER; det += 6) {
    if (mUsedDetectors[det]) {
      int sector = mGeo->getSector(det);
      int stack = mGeo->getStack(det);

      json digitsJson = {
        {"evid", iEvent},
        {"lyrs", json::array()}};

      for (int iLayer = 0; iLayer < 6; iLayer++) {
        json layerJson = {
          {"lyr", iLayer},
          {"pads", json::array()}};

        for (int iDigit = trigRec.getFirstDigit(); iDigit < trigRec.getFirstDigit() + trigRec.getNumberOfDigits(); ++iDigit) {
          Digit digit = digits[iDigit];

          int detector = digit.getDetector();

          if (detector == det + iLayer) {
            // Digits are in stack with track
            int row = digit.getPadRow();
            int col = digit.getPadCol();

            json padJson = {
              {"row", row},
              {"col", col},
              {"tbins", json::array()}};

            for (auto adc : digit.getADC()) {
              padJson["tbins"].push_back(adc);
            }
            layerJson["pads"].push_back(padJson);
          }
        }
        digitsJson["lyrs"].push_back(layerJson);
      }
      std::ofstream digitsOut(Form("../alice-trd-event-display/data/o2/E%d.%d.%d.json", iEvent, sector, stack));
      digitsOut << digitsJson.dump(4);
    }
  }
} // namespace trd

void TRDEventDisplayFeedSpec::run(o2::framework::ProcessingContext& pc)
{
  LOG(info) << "Running event display feed...";

  json jsonData = json::array();

  auto tracks = pc.inputs().get<gsl::span<TrackTRD>>("trdtracks");
  auto tracklets = pc.inputs().get<gsl::span<Tracklet64>>("trdtracklets");
  auto digits = pc.inputs().get<gsl::span<Digit>>("trddigits");

  mTrigRecs = pc.inputs().get<gsl::span<TriggerRecord>>("trdtriggerrec");
  auto trackTrigRecs = pc.inputs().get<gsl::span<TrackTriggerRecord>>("tracktriggerrec");

  int nEvents = std::min((int)mTrigRecs.size(), mNeventsMax);

  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    const auto& trackTrigRec = trackTrigRecs[iEvent];

    if (trackTrigRec.getNumberOfTracks() == 0) {
      continue;
    }

    mUsedDetectors.reset();

    // Get run parameters
    o2::base::Propagator::initFieldFromGRP();
    auto prop = o2::base::Propagator::Instance();
    mBz = prop->getNominalBz();

    auto field = o2::field::MagneticField::createNominalField(mBz);
    double beamEnergy = field->getBeamEnergy();
    std::string beamType = field->getBeamTypeText();

    auto grp = o2::parameters::GRPObject::loadFrom();
    auto triggers = grp->getDetsTrigger();
    auto triggerNames = o2::detectors::DetID::getNames(triggers);

    json eventJson = {
      {"id", Form("E%d", iEvent)},
      {"i", {{"be", beamEnergy}, {"bt", beamType}, {"ft", triggerNames}}},
      {"tracks", json::array()},
      {"trklts", json::array()}};

    eventJson["tracks"] = getTracksJson(tracks, tracklets, trackTrigRecs, iEvent);

    eventJson["trklts"] = getTrackletsJson(tracklets, iEvent);

    writeDigits(digits, iEvent);

    jsonData.push_back(eventJson);
  }
  std::ofstream jsScriptOut(Form("../alice-trd-event-display/data/o2/script.js"));
  // Path to data file for alicetrd.web: lxplus.cern.ch:/eos/project/a/alice-trd/www/eventdisplay/data/o2/script.js

  jsScriptOut << "function getDigitsLoadUrl(eventNo, sector, stack) { return `"
              << "data/o2/${eventNo}.${sector}.${stack}.json`; }"
              << std::endl
              << std::endl
              << "function getData() {\n\treturn "
              << jsonData.dump(4) << "}";
}

o2::framework::DataProcessorSpec getTRDEventDisplayFeedSpec(int nEventsMax)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("trdtracks", "TRD", "MATCH_ITSTPC", 0, Lifetime::Timeframe);
  inputs.emplace_back("trdtracklets", "TRD", "TRACKLETS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trddigits", ConcreteDataTypeMatcher{o2::header::gDataOriginTRD, "DIGITS"}, Lifetime::Timeframe);
  inputs.emplace_back("trdtriggerrec", "TRD", "TRKTRGRD", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracktriggerrec", "TRD", "TRGREC_ITSTPC", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TRDEVENTDISPLAYFEED",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDEventDisplayFeedSpec>(nEventsMax)},
    Options{}};
}

} // namespace trd
} //end namespace o2
