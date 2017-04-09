///
/// @file  Baseline.cxx
/// @author Jens Wiechula, Jens.Wiechula@cern.ch
///

#include "TPCSimulation/Baseline.h"

#include "TPCBase/PadSecPos.h"

using namespace o2::TPC;

float Baseline::getNoise(const PadSecPos& padSecPos)
{
  switch (mBaselineType) {
    case BaselineType::Random: {
      return getRandomNoise();
      break;
    }
    case BaselineType::PseudoRealistic: {
      return getPseudoRealisticNoise(padSecPos);
      break;
    }
    case BaselineType::DataBase: {
      return getNoiseFromDataBase(padSecPos);
      break;
    }
  }
}

//______________________________________________________________________________
float_v Baseline::getNoiseVc(const PadSecPos& padSecPos)
{
  switch (mBaselineType) {
    case BaselineType::Random: {
      return getRandomNoiseVc();
      break;
    }
    case BaselineType::PseudoRealistic: {
      return getPseudoRealisticNoiseVc(padSecPos);
      break;
    }
    case BaselineType::DataBase: {
      return getNoiseFromDataBaseVc(padSecPos);
      break;
    }
  }
}

//______________________________________________________________________________
float Baseline::getRandomNoise()
{
  return mRandomNoiseRing.getNextValue()*mMeanNoise;
}

//______________________________________________________________________________
float Baseline::getPseudoRealisticNoise(const PadSecPos& padSecPos)
{
}

//______________________________________________________________________________
float Baseline::getNoiseFromDataBase(const PadSecPos& padSecPos)
{
}

//______________________________________________________________________________
float_v Baseline::getRandomNoiseVc()
{
  return mRandomNoiseRing.getNextValueVc()*mMeanNoise;
}

//______________________________________________________________________________
float_v Baseline::getPseudoRealisticNoiseVc(const PadSecPos& padSecPos)
{
}

//______________________________________________________________________________
float_v Baseline::getNoiseFromDataBaseVc(const PadSecPos& padSecPos)
{
}

//______________________________________________________________________________
float Baseline::getRandomPedestal(const PadSecPos& padSecPos)
{
}

//______________________________________________________________________________
float Baseline::getPseudoRealisticPedestal(const PadSecPos& padSecPos)
{
}

//______________________________________________________________________________
float Baseline::getPedestalFromDataBase(const PadSecPos& padSecPos)
{
}


