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
#include <algorithm>
#include <iostream>
#include <cmath>
#include <TMath.h> // for TMath::Pi() - to be removed once we switch to c++20
#include "EMCALBase/NonlinearityHandler.h"

using namespace o2::emcal;

NonlinearityHandler::NonlinearityHandler(NonlinType_t nonlintype) : mNonlinearyFunction(nonlintype)
{
  initParams();
}

void NonlinearityHandler::initParams()
{
  std::fill(mNonlinearityParam.begin(), mNonlinearityParam.end(), 0);
  switch (mNonlinearyFunction) {
    case NonlinType_t::MC_TESTBEAM_FINAL:
      mNonlinearityParam[0] = 1.09357;
      mNonlinearityParam[1] = 0.0192266;
      mNonlinearityParam[2] = 0.291993;
      mNonlinearityParam[3] = 370.927;
      mNonlinearityParam[4] = 694.656;
      break;
    case NonlinType_t::MC_PI0:
      mNonlinearityParam[0] = 1.014;
      mNonlinearityParam[1] = -0.03329;
      mNonlinearityParam[2] = -0.3853;
      mNonlinearityParam[3] = 0.5423;
      mNonlinearityParam[4] = -0.4335;
      break;
    case NonlinType_t::MC_PI0_V2:
      mNonlinearityParam[0] = 3.11111e-02;
      mNonlinearityParam[1] = -5.71666e-02;
      mNonlinearityParam[2] = 5.67995e-01;
      break;
    case NonlinType_t::MC_PI0_V3:
      mNonlinearityParam[0] = 9.81039e-01;
      mNonlinearityParam[1] = 1.13508e-01;
      mNonlinearityParam[2] = 1.00173e+00;
      mNonlinearityParam[3] = 9.67998e-02;
      mNonlinearityParam[4] = 2.19381e+02;
      mNonlinearityParam[5] = 6.31604e+01;
      mNonlinearityParam[6] = 1;
      break;
    case NonlinType_t::MC_PI0_V5:
      mNonlinearityParam[0] = 1.0;
      mNonlinearityParam[1] = 6.64778e-02;
      mNonlinearityParam[2] = 1.570;
      mNonlinearityParam[3] = 9.67998e-02;
      mNonlinearityParam[4] = 2.19381e+02;
      mNonlinearityParam[5] = 6.31604e+01;
      mNonlinearityParam[6] = 1.01286;
      break;
    case NonlinType_t::MC_PI0_V6:
      mNonlinearityParam[0] = 1.0;
      mNonlinearityParam[1] = 0.0797873;
      mNonlinearityParam[2] = 1.68322;
      mNonlinearityParam[3] = 0.0806098;
      mNonlinearityParam[4] = 244.586;
      mNonlinearityParam[5] = 116.938;
      mNonlinearityParam[6] = 1.00437;
      break;
    case NonlinType_t::DATA_TESTBEAM_CORRECTED:
      mNonlinearityParam[0] = 0.99078;
      mNonlinearityParam[1] = 0.161499;
      mNonlinearityParam[2] = 0.655166;
      mNonlinearityParam[3] = 0.134101;
      mNonlinearityParam[4] = 163.282;
      mNonlinearityParam[5] = 23.6904;
      mNonlinearityParam[6] = 0.978;
      break;
    case NonlinType_t::DATA_TESTBEAM_CORRECTED_V2:
      // Parameters until November 2015, use now kBeamTestCorrectedv3
      mNonlinearityParam[0] = 0.983504;
      mNonlinearityParam[1] = 0.210106;
      mNonlinearityParam[2] = 0.897274;
      mNonlinearityParam[3] = 0.0829064;
      mNonlinearityParam[4] = 152.299;
      mNonlinearityParam[5] = 31.5028;
      mNonlinearityParam[6] = 0.968;
      break;
    case NonlinType_t::DATA_TESTBEAM_CORRECTED_V3:
      // New parametrization of kBeamTestCorrected
      // excluding point at 0.5 GeV from Beam Test Data
      // https://indico.cern.ch/event/438805/contribution/1/attachments/1145354/1641875/emcalPi027August2015.pdf

      mNonlinearityParam[0] = 0.976941;
      mNonlinearityParam[1] = 0.162310;
      mNonlinearityParam[2] = 1.08689;
      mNonlinearityParam[3] = 0.0819592;
      mNonlinearityParam[4] = 152.338;
      mNonlinearityParam[5] = 30.9594;
      mNonlinearityParam[6] = 0.9615;
      break;

    case NonlinType_t::DATA_TESTBEAM_CORRECTED_V4:
      // New parametrization of kBeamTestCorrected,
      // fitting new points for E>100 GeV.
      // I should have same performance as v3 in the low energies
      // See EMCal meeting 21/09/2018 slides
      // https://indico.cern.ch/event/759154/contributions/3148448/attachments/1721042/2778585/nonLinearityUpdate.pdf
      //  and jira ticket EMCAL-190

      mNonlinearityParam[0] = 0.9892;
      mNonlinearityParam[1] = 0.1976;
      mNonlinearityParam[2] = 0.865;
      mNonlinearityParam[3] = 0.06775;
      mNonlinearityParam[4] = 156.6;
      mNonlinearityParam[5] = 47.18;
      mNonlinearityParam[6] = 0.97;
      break;

    case NonlinType_t::DATA_TESTBEAM_SHAPER:
    case NonlinType_t::DATA_TESTBEAM_SHAPER_WOSCALE:
      mNonlinearityParam[0] = 1.91897;
      mNonlinearityParam[1] = 0.0264988;
      mNonlinearityParam[2] = 0.965663;
      mNonlinearityParam[3] = -187.501;
      mNonlinearityParam[4] = 2762.51;
      break;

    default:
      break;
  }
  if (mNonlinearyFunction == NonlinType_t::DATA_TESTBEAM_SHAPER) {
    mApplyScaleCorrection = true;
  }
}
double NonlinearityHandler::getCorrectedClusterEnergy(double energy) const
{
  double correctedEnergy = energy;
  switch (mNonlinearyFunction) {
    case NonlinType_t::MC_PI0:
      correctedEnergy = evaluatePi0MC(energy);
      break;
    case NonlinType_t::MC_PI0_V2:
      correctedEnergy = evaluatePi0MCv2(energy);
    case NonlinType_t::MC_PI0_V3:
    case NonlinType_t::MC_PI0_V5:
    case NonlinType_t::MC_PI0_V6:
    case NonlinType_t::DATA_TESTBEAM_CORRECTED:
    case NonlinType_t::DATA_TESTBEAM_CORRECTED_V2:
    case NonlinType_t::DATA_TESTBEAM_CORRECTED_V3:
    case NonlinType_t::DATA_TESTBEAM_CORRECTED_V4:
      correctedEnergy = evaluateTestbeamCorrected(energy);
      break;
    case NonlinType_t::DATA_TESTBEAM_SHAPER:
    case NonlinType_t::DATA_TESTBEAM_SHAPER_WOSCALE:
    case NonlinType_t::MC_TESTBEAM_FINAL:
      correctedEnergy = evaluateTestbeamShaper(energy);
      break;
    default:
      throw UninitException();
  }
  if (mApplyScaleCorrection) {
    correctedEnergy *= 1.0505;
  }
  return correctedEnergy;
}

double NonlinearityHandler::evaluateTestbeamShaper(double energy) const
{
  return energy / (1.00 * (mNonlinearityParam[0] + mNonlinearityParam[1] * std::log(energy)) / (1 + (mNonlinearityParam[2] * std::exp((energy - mNonlinearityParam[3]) / mNonlinearityParam[4]))));
}

double NonlinearityHandler::evaluateTestbeamCorrected(double energy) const
{
  return energy * mNonlinearityParam[6] / (mNonlinearityParam[0] * (1. / (1. + mNonlinearityParam[1] * std::exp(-energy / mNonlinearityParam[2])) * 1. / (1. + mNonlinearityParam[3] * std::exp((energy - mNonlinearityParam[4]) / mNonlinearityParam[5]))));
}

double NonlinearityHandler::evaluatePi0MC(double energy) const
{
  return energy * (mNonlinearityParam[0] * std::exp(-mNonlinearityParam[1] / energy)) +
         ((mNonlinearityParam[2] / (mNonlinearityParam[3] * 2. * TMath::Pi()) *
           std::exp(-(energy - mNonlinearityParam[4]) * (energy - mNonlinearityParam[4]) / (2. * mNonlinearityParam[3] * mNonlinearityParam[3]))));
}

double NonlinearityHandler::evaluatePi0MCv2(double energy) const
{
  return energy * mNonlinearityParam[0] / TMath::Power(energy + mNonlinearityParam[1], mNonlinearityParam[2]) + 1;
}

double NonlinearityHandler::evaluateShaperCorrectionCellEnergy(double energy, double ecalibHG)
{
  if (energy < 40) {
    return energy * 16.3 / 16;
  }
  constexpr std::array<double, 8> par = {{1, 29.8279, 0.607704, 0.00164896, -2.28595e-06, -8.54664e-10, 5.50191e-12, -3.28098e-15}};
  double x = par[0] * energy / ecalibHG / 16 / 0.0162;

  double res = par[1];
  res += par[2] * x;
  res += par[3] * x * x;
  res += par[4] * x * x * x;
  res += par[5] * x * x * x * x;
  res += par[6] * x * x * x * x * x;
  res += par[7] * x * x * x * x * x * x;

  return ecalibHG * 16.3 * res * 0.0162;
}

void NonlinearityHandler::printStream(std::ostream& stream) const
{
  stream << "Nonlinearity function: " << getNonlinName(mNonlinearyFunction)
         << "(Parameters:";
  bool first = true;
  for (auto& param : mNonlinearityParam) {
    if (first) {
      first = false;
    } else {
      stream << ",";
    }
    stream << " " << param;
  }
  stream << ")";
}

NonlinearityHandler::NonlinType_t NonlinearityHandler::getNonlinType(const std::string_view name)
{
  using NLType = NonlinearityHandler::NonlinType_t;
  if (name == "MC_Pi0") {
    return NLType::MC_PI0;
  }
  if (name == "MC_Pi0_v2") {
    return NLType::MC_PI0_V2;
  }
  if (name == "MC_Pi0_v3") {
    return NLType::MC_PI0_V3;
  }
  if (name == "MC_Pi0_v5") {
    return NLType::MC_PI0_V5;
  }
  if (name == "MC_Pi0_v6") {
    return NLType::MC_PI0_V6;
  }
  if (name == "MC_TestbeamFinal") {
    return NLType::MC_TESTBEAM_FINAL;
  }
  if (name == "DATA_BeamTestCorrected") {
    return NLType::DATA_TESTBEAM_CORRECTED;
  }
  if (name == "DATA_BeamTestCorrected_v2") {
    return NLType::DATA_TESTBEAM_CORRECTED_V2;
  }
  if (name == "DATA_BeamTestCorrected_v3") {
    return NLType::DATA_TESTBEAM_CORRECTED_V3;
  }
  if (name == "DATA_BeamTestCorrected_v4") {
    return NLType::DATA_TESTBEAM_CORRECTED_V4;
  }
  if (name == "DATA_TestbeamFinal") {
    return NLType::DATA_TESTBEAM_SHAPER;
  }
  if (name == "DATA_TestbeamFinal_NoScale") {
    return NLType::DATA_TESTBEAM_SHAPER_WOSCALE;
  }
  return NLType::UNKNOWN;
}

const char* NonlinearityHandler::getNonlinName(NonlinearityHandler::NonlinType_t nonlin)
{
  using NLType = NonlinearityHandler::NonlinType_t;
  switch (nonlin) {
    case NLType::MC_PI0:
      return "MC_Pi0";
    case NLType::MC_PI0_V2:
      return "MC_Pi0_v2";
    case NLType::MC_PI0_V3:
      return "MC_Pi0_v3";
    case NLType::MC_PI0_V5:
      return "MC_Pi0_v5";
    case NLType::MC_PI0_V6:
      return "MC_Pi0_v6";
    case NLType::MC_TESTBEAM_FINAL:
      return "MC_TestbeamFinal";
    case NLType::DATA_TESTBEAM_CORRECTED:
      return "DATA_BeamTestCorrected";
    case NLType::DATA_TESTBEAM_CORRECTED_V2:
      return "DATA_BeamTestCorrected_v2";
    case NLType::DATA_TESTBEAM_CORRECTED_V3:
      return "DATA_BeamTestCorrected_v3";
    case NLType::DATA_TESTBEAM_CORRECTED_V4:
      return "DATA_BeamTestCorrected_v4";
    case NLType::DATA_TESTBEAM_SHAPER:
      return "DATA_TestbeamFinal";
    case NLType::DATA_TESTBEAM_SHAPER_WOSCALE:
      return "DATA_TestbeamFinal_NoScale";
    case NLType::UNKNOWN:
      return "Unknown";
    default:
      return "";
  }
}

NonlinearityHandler& NonlinearityFactory::getNonlinearity(NonlinearityHandler::NonlinType_t nonlintype)
{
  auto found = mHandlers.find(nonlintype);
  if (found != mHandlers.end()) {
    return found->second;
  }
  auto [insert_result, insert_status] = mHandlers.try_emplace(nonlintype, NonlinearityHandler(nonlintype));
  if (insert_status) {
    return insert_result->second;
  }
  throw NonlinInitError();
}

NonlinearityHandler& NonlinearityFactory::getNonlinearity(const std::string_view nonname)
{
  return getNonlinearity(getNonlinType(nonname));
}

NonlinearityHandler::NonlinType_t NonlinearityFactory::getNonlinType(const std::string_view nonlinName) const
{
  auto found = mNonlinNames.find(static_cast<std::string>(nonlinName));
  if (found != mNonlinNames.end()) {
    return found->second;
  }
  throw NonlinearityFactory::FunctionNotFoundExcpetion(nonlinName);
}

void NonlinearityFactory::initNonlinNames()
{
  using NLType = NonlinearityHandler::NonlinType_t;
  constexpr std::array<NLType, 12> nonlintypes = {{NLType::MC_PI0,
                                                   NLType::MC_PI0_V2,
                                                   NLType::MC_PI0_V3,
                                                   NLType::MC_PI0_V5,
                                                   NLType::MC_PI0_V6,
                                                   NLType::MC_TESTBEAM_FINAL,
                                                   NLType::DATA_TESTBEAM_CORRECTED,
                                                   NLType::DATA_TESTBEAM_CORRECTED_V2,
                                                   NLType::DATA_TESTBEAM_CORRECTED_V3,
                                                   NLType::DATA_TESTBEAM_CORRECTED_V4,
                                                   NLType::DATA_TESTBEAM_SHAPER,
                                                   NLType::DATA_TESTBEAM_SHAPER_WOSCALE

  }};
  for (auto nonlin : nonlintypes) {
    mNonlinNames[NonlinearityHandler::getNonlinName(nonlin)] = nonlin;
  }
}

std::ostream& o2::emcal::operator<<(std::ostream& in, const NonlinearityHandler& handler)
{
  handler.printStream(in);
  return in;
}