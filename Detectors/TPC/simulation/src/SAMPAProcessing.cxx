/// \file SAMPAProcessing.cxx
/// \brief Implementation of the SAMPA response
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCSimulation/Digitizer.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "FairLogger.h"

using namespace o2::TPC;

SAMPAProcessing::SAMPAProcessing()
  : mSaturationSpline()
{
  importSaturationCurve("SAMPA_saturation.dat");
}

SAMPAProcessing::~SAMPAProcessing()
= default;

bool SAMPAProcessing::importSaturationCurve(std::string file)
{
  std::string inputDir;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if(aliceO2env) {
    inputDir = aliceO2env;
  }
  inputDir += "/share/Detectors/TPC/files/";

  std::ifstream saturationFile(inputDir + file, std::ifstream::in);
  if (!saturationFile) {
    LOG(FATAL) << "TPC::SAMPAProcessing - Input file '" << inputDir + file << "' does not exist! No SAMPA saturation curve loaded!" << FairLogger::endl;
    return false;
  }
  std::vector<std::pair<float, float>> saturation;
  for(std::string line; std::getline(saturationFile, line);) {
    float x, y;
    std::istringstream is(line);
    while (is >> x >> y) {
      saturation.emplace_back(x, y);
    }
  }
  const int nPoints = saturation.size();
  double xSat[nPoints], ySat[nPoints];
  for (int i = 0; i < nPoints; ++i) {
    xSat[i] = saturation[i].first;
    ySat[i] = saturation[i].second;
  }

  mSaturationSpline = std::unique_ptr<TSpline3>(new TSpline3("SAMPA saturation", xSat, ySat, nPoints, "b2e2", 0, 0));
  return true;
}

void SAMPAProcessing::getShapedSignal(float ADCsignal, float driftTime, std::array<float, mNShapedPoints>& signalArray)
{
  float timeBinTime = Digitizer::getTimeBinTime(driftTime);
  float offset = driftTime - timeBinTime;
  signalArray.fill(0);
  for (float bin = 0; bin < mNShapedPoints; bin += Vc::float_v::Size) {
    Vc::float_v binvector;
    for (int i = 0; i < Vc::float_v::Size; ++i) {
      binvector[i] = bin + i;
    }
    Vc::float_v time = timeBinTime + binvector * ZBINWIDTH;
    Vc::float_v signal = getGamma4(time, Vc::float_v(timeBinTime+offset), Vc::float_v(ADCsignal));
    for (int i = 0; i < Vc::float_v::Size; ++i) {
      signalArray[bin+i] = signal[i];
    }
  }
}
