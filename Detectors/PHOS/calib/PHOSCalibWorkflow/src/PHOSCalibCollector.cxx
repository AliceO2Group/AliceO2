// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSCalibWorkflow/PHOSCalibCollector.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ControlService.h"

#include "FairLogger.h"
#include <fstream> // std::ifstream

using namespace o2::phos;

void PHOSCalibCollector::init(o2::framework::InitContext& ic)
{

  mEvent = 0;
  //Create output histograms
  const int nChannels = 14336; //4 full modules
  const int nMass = 150.;
  float massMax = 0.3;
  //First create all histograms then get direct pointers
  mHistos.emplace_back("hReInvMassPerCell", "Real inv. mass per cell", nChannels, 0., nChannels, nMass, 0., massMax);
  mHistos.emplace_back("hMiInvMassPerCell", "Mixed inv. mass per cell", nChannels, 0., nChannels, nMass, 0., massMax);

  int npt = 45;
  float xpt[46] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.4, 3.8, 4.2, 4.6, 5.,
                   5.5, 6., 6.5, 7., 7.5, 8., 9., 10., 12., 14., 16., 18., 20., 24., 28., 32., 36., 40., 50., 55., 60.};
  float massbins[nMass + 1];
  for (int i = 0; i <= nMass; i++) {
    massbins[i] = i * massMax / nMass;
  }
  mHistos.emplace_back("hReInvMassNonlin", "Real inv. mass vs Eclu", nMass, massbins, npt, xpt);
  mHistos.emplace_back("hMiInvMassNonlin", "Mixed inv. mass vs Eclu", nMass, massbins, npt, xpt);

  const int nTime = 200;
  float timeMin = -100.e-9;
  float timeMax = 100.e-9;
  mHistos.emplace_back("hTimeHGPerCell", "time per cell, high gain", nChannels, 0., nChannels, nTime, timeMin, timeMax);
  mHistos.emplace_back("hTimeLGPerCell", "time per cell, low gain", nChannels, 0., nChannels, nTime, timeMin, timeMax);
  float timebins[nTime + 1];
  for (int i = 0; i <= nTime; i++) {
    timebins[i] = timeMin + i * timeMax / nTime;
  }
  mHistos.emplace_back("hTimeHGSlewing", "time vs E, high gain", nTime, timebins, npt, xpt);
  mHistos.emplace_back("hTimeLGSlewing", "time vs E, low gain", nTime, timebins, npt, xpt);

  if (mMode != 2) {
    TFile fcalib(mfilenameCalib.data(), "READ");
    if (fcalib.IsOpen()) {
      CalibParams* cp = nullptr;
      fcalib.GetObject("PHOSCalibration", cp);
      mCalibParams.reset(cp);
    } else {
      LOG(ERROR) << "can not read calibration <PHOSCalibration> from file " << mfilenameCalib;
    }
    fcalib.Close();
  }

  //TODO: configure reading bad map from file
  // this is special bad map for calibration
  mBadMap.reset(new BadChannelMap());

  mGeom = Geometry::GetInstance("Run3");
}

void PHOSCalibCollector::run(o2::framework::ProcessingContext& pc)
{

  //select possible options
  switch (mMode) {
    case 0: // Read new data
      scanClusters(pc);
      writeOutputs();
      break;
    case 1: // Read and re-calibrate stored trees
      readDigits();
      writeOutputs();
      break;
    case 2: //Calculate calibration from stored histograms
      calculateCalibrations();
      compareCalib();
      sendOutput(pc.outputs());
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      break;
  }
}

void PHOSCalibCollector::scanClusters(o2::framework::ProcessingContext& pc)
{

  //  auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime; // is this the timestamp of the current TF?

  auto clusters = pc.inputs().get<std::vector<o2::phos::FullCluster>>("clusters");
  auto cluTR = pc.inputs().get<gsl::span<o2::phos::TriggerRecord>>("cluTR");
  LOG(INFO) << "Processing TF with " << clusters.size() << " clusters and " << cluTR.size() << " TriggerRecords";

  for (auto& tr : cluTR) {

    //Mark new event
    //First goes new event marker + BC (16 bit), next word orbit (32 bit)
    EventHeader h = {0};
    h.mMarker = 16383;
    h.mBC = tr.getBCData().bc;
    mDigits.push_back(h.mDataWord);
    mDigits.push_back(tr.getBCData().orbit);

    int iclu = 0;
    int firstCluInEvent = tr.getFirstEntry();
    int lastCluInEvent = firstCluInEvent + tr.getNumberOfObjects();

    mBuffer->startNewEvent(); // mark stored clusters to be used for Mixing
    for (int i = firstCluInEvent; i < lastCluInEvent; i++) {
      const FullCluster& clu = clusters[i];

      fillTimeMassHisto(clu);
      bool isGood = checkCluster(clu);

      auto cluList = clu.getElementList();
      for (auto ce = cluList->begin(); ce != cluList->end(); ce++) {
        short absId = ce->absId;
        //Fill cells from cluster for next iterations
        short adcCounts = ce->energy / mCalibParams->getGain(absId);
        // Need to chale LG gain too to fit dynamic range
        if (!ce->isHG) {
          adcCounts /= mCalibParams->getHGLGRatio(absId);
        }
        CalibDigit d = {0};
        d.mAddress = absId;
        d.mAdcAmp = adcCounts;
        d.mHgLg = ce->isHG;
        d.mBadChannel = isGood;
        d.mCluster = (i - firstCluInEvent) % kMaxCluInEvent;
        mDigits.push_back(d.mDataWord);
        if (i - firstCluInEvent > kMaxCluInEvent) {
          //Normally this is not critical as indexes are used "locally", i.e. are compared to previous/next
          LOG(INFO) << "Too many clusters per event:" << i - firstCluInEvent << ", apply more strict selection; clusters with same indexes will appear";
        }
      }
    }
  }
}

void PHOSCalibCollector::readDigits()
{
  // open files from the list
  // for each file read digits, calibrate them and form clusters
  // fill inv mass and time (for check) histograms
  std::ifstream ifs(mdigitsfilelist, std::ifstream::in);
  if (!ifs.is_open()) {
    LOG(ERROR) << "can not open file " << mdigitsfilelist;
    return;
  }

  std::string filename;
  while (ifs >> filename) {
    TFile f(filename.data(), "READ");
    if (!f.IsOpen()) {
      LOG(ERROR) << "can not read file " << filename;
      continue;
    }
    std::vector<uint32_t>* digits;
    f.GetObject("Digits", digits);

    std::vector<uint32_t>::const_iterator digIt = digits->cbegin();
    std::vector<uint32_t>::const_iterator digEnd = digits->cend();
    FullCluster clu;
    bool isNextNewEvent;
    mBuffer->startNewEvent(); // mark stored clusters to be used for Mixing
    while (nextCluster(digIt, digEnd, clu, isNextNewEvent)) {

      fillTimeMassHisto(clu);
      if (isNextNewEvent) {
        mBuffer->startNewEvent(); // mark stored clusters to be used for Mixing
      }
    }
    delete digits;
    f.Close();
  }
}

bool PHOSCalibCollector::nextCluster(std::vector<uint32_t>::const_iterator digitIt, std::vector<uint32_t>::const_iterator digitEnd,
                                     FullCluster& clu, bool& isNextNewEvent)
{
  //Scan digits belonging to cluster
  // return true if cluster read
  isNextNewEvent = false;
  clu.reset();
  int cluIndex = -1;
  while (digitIt != digitEnd) {
    CalibDigit d = {*digitIt};
    if (d.mAddress == 16383) {         //impossible address, marker of new event
      if (clu.getMultiplicity() > 0) { //already read cluster: calculate its parameters; do not go into next event
        isNextNewEvent = true;
        break;
      }
      //If just started cluster, read new event. Two first words event header
      EventHeader h = {*digitIt};
      mEvBC = h.mBC; //current event BC
      digitIt++;
      mEvOrbit = *digitIt; //current event orbit
      digitIt++;
      continue;
    }
    if (cluIndex == -1) { //start new cluster
      cluIndex = d.mCluster;
    } else {
      if (cluIndex != d.mCluster) { //digit belongs to text cluster: all digits from current read
        break;
      }
    }
    short absId = d.mAddress;
    float energy = d.mAdcAmp * mCalibParams->getGain(absId);
    if (!d.mHgLg) {
      energy *= mCalibParams->getHGLGRatio(absId);
    }
    clu.addDigit(absId, energy, 0., 0, 1.);
    digitIt++;
  }
  if (digitIt == digitEnd && clu.getMultiplicity() == 0) {
    return false;
  }
  //Evaluate clu parameters
  clu.purify();
  clu.evalAll();
  return true;
}

//Write
void PHOSCalibCollector::writeOutputs()
{
  //Write digits only in first scan
  if (mMode == 0) {
    TFile fout(mdigitsfilename.data(), "recreate");
    fout.WriteObjectAny(&mDigits, "std::vector<uint32_t", "Digits");
    fout.Close();
  }

  // in all cases write inv mass distributions
  TFile fHistoOut(mhistosfilename.data(), "recreate");
  for (auto h : mHistos) {
    h.Write();
  }
  fHistoOut.Close();
}

void PHOSCalibCollector::fillTimeMassHisto(const FullCluster& clu)
{
  // Fill time distributions only for cells in cluster
  if (mMode == 0) {
    auto cluList = clu.getElementList();
    for (auto ce = cluList->begin(); ce != cluList->end(); ce++) {
      short absId = ce->absId;
      if (ce->isHG) {
        if (ce->energy > mEminHGTime) {
          mHistos[kTimeHGPerCell].Fill(absId, ce->time);
        }
        mHistos[kTimeHGSlewing].Fill(ce->time, ce->energy);
      } else {
        if (ce->energy > mEminLGTime) {
          mHistos[kTimeLGPerCell].Fill(absId, ce->time);
        }
        mHistos[kTimeLGSlewing].Fill(ce->time, ce->energy);
      }
    }
  }

  //Real and Mixed inv mass distributions
  // prepare TLorentsVector
  float posX, posZ;
  clu.getLocalPosition(posX, posZ);
  TVector3 vec3;
  mGeom->local2Global(clu.module(), posX, posZ, vec3);
  vec3 -= mVertex;
  float e = clu.getEnergy();
  short absId;
  mGeom->relPosToAbsId(clu.module(), posX, posZ, absId);

  TLorentzVector v(vec3.X() * e, vec3.Y() * e, vec3.Z() * e, e);
  // Fill calibration histograms for all cells, even bad, but partners in inv, mass should be good
  bool isGood = checkCluster(clu);
  for (short ip = mBuffer->size(); --ip;) {
    const TLorentzVector& vp = mBuffer->getEntry(ip);
    TLorentzVector sum = v + vp;
    if (mBuffer->isCurrentEvent(ip)) { //same (real) event
      if (isGood) {
        mHistos[kReInvMassNonlin].Fill(e, sum.M());
      }
      if (sum.Pt() > mPtMin) {
        mHistos[kReInvMassPerCell].Fill(absId, sum.M());
      }
    } else { //Mixed
      if (isGood) {
        mHistos[kMiInvMassNonlin].Fill(e, sum.M());
      }
      if (sum.Pt() > mPtMin) {
        mHistos[kMiInvMassPerCell].Fill(absId, sum.M());
      }
    }
  }

  //Add to list ot partners only if cluster is good
  if (isGood) {
    mBuffer->addEntry(v);
  }
}

bool PHOSCalibCollector::checkCluster(const FullCluster& clu)
{
  //First check BadMap
  float posX, posZ;
  clu.getLocalPosition(posX, posZ);
  short absId;
  Geometry::relPosToAbsId(clu.module(), posX, posZ, absId);
  if (!mBadMap->isChannelGood(absId)) {
    return false;
  }

  return (clu.getEnergy() > 0.3 && clu.getMultiplicity() > 1);
}

void PHOSCalibCollector::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  //Write Filledhistograms and estimate mean number of entries
  switch (mMode) {
    case 0: // Read new data
      writeOutputs();
      break;
    case 1: // Read and re-calibrate stored trees
      writeOutputs();
      break;
    case 2: //Calculate calibration from stored histograms
      // do nothing, already sent
      break;
  }
}

void PHOSCalibCollector::sendOutput(DataAllocator& output)
{
  // If calibration OK, send calibration to CCDB
  // and difference to last iteration to QC
  //....
}

o2::framework::DataProcessorSpec o2::phos::getPHOSCalibCollectorDeviceSpec(int mode)
{

  std::vector<OutputSpec> outputs;
  if (mode == 2) { //fit of inv masses: gain calculation: send to CCDB and QC
    outputs.emplace_back(o2::header::gDataOriginPHS, "COLLECTEDINFO", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginPHS, "ENTRIESCH", 0, Lifetime::Timeframe);
  }

  std::vector<InputSpec> inputs;
  if (mode == 0) {
    inputs.emplace_back("clusters", "PHS", "CLUSTERS");
    inputs.emplace_back("cluTR", "PHS", "CLUSTERTRIGRECS");
  }

  return DataProcessorSpec{
    "calib-phoscalib-collector",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PHOSCalibCollector>(mode)},
    Options{
      {"inputFileList", VariantType::String, "./calibDigits.txt", {"File with list of digit files to be scanned"}},
      {"outputDigitDir", VariantType::String, "./", {"directory to write filtered Digits"}},
      {"outputHistoDir", VariantType::String, "./", {"directory to write inv. mass histograms"}},
      {"forceCCDBUpdate", VariantType::Bool, false, {"force updating CCDB bypassing quality check"}}}};
}
