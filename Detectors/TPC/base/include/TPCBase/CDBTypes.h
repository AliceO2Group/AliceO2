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

/// \file CDBTypes.h
/// \brief CDB Type definitions for TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_CDBTypes_H_
#define AliceO2_TPC_CDBTypes_H_

#include <unordered_map>
#include <string>

namespace o2::tpc
{

/// Calibration and parameter types for CCDB
enum class CDBType {
  CalPedestal,         ///< Pedestal calibration
  CalNoise,            ///< Noise calibration
  CalPedestalNoise,    ///< Pedestal and Noise calibration
  CalPulser,           ///< Pulser calibration
  CalCE,               ///< Laser CE calibration
  CalPadGainFull,      ///< Full pad gain calibration
  CalPadGainResidual,  ///< ResidualpPad gain calibration (e.g. from tracks)
  CalLaserTracks,      ///< Laser track calibration data
  CalVDriftTgl,        ///< ITS-TPC difTgl vdrift calibration
  CalTimeGain,         ///< Gain variation over time
  CalGas,              ///< DCS gas measurements
  CalTemperature,      ///< DCS temperature measurements
  CalHV,               ///< DCS HV measurements
  CalTopologyGain,     ///< Q cluster topology correction
                       ///
  ConfigFEEPad,        ///< FEE pad-by-pad configuration map
  ConfigFEE,           ///< FEE configuration map for each tag
  ConfigRunInfo,       ///< FEE run information (run -> tag)
                       ///
  ParDetector,         ///< Parameter for Detector
  ParElectronics,      ///< Parameter for Electronics
  ParGas,              ///< Parameter for Gas
  ParGEM,              ///< Parameter for GEM
                       ///
  CalIDC0A,            ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  CalIDC0C,            ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  CalIDC1A,            ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  CalIDC1C,            ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  CalIDCDeltaA,        ///< \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  CalIDCDeltaC,        ///< \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  CalIDCFourierA,      ///< Fourier coefficients of CalIDC1
  CalIDCFourierC,      ///< Fourier coefficients of CalIDC1
  CalIDCPadStatusMapA, ///< Status map of the pads (dead etc. obatined from CalIDC0)
  CalIDCPadStatusMapC, ///< Status map of the pads (dead etc. obatined from CalIDC0)
  CalIDCGroupingParA,  ///< Parameters which were used for the averaging of the CalIDCDelta
  CalIDCGroupingParC,  ///< Parameters which were used for the averaging of the CalIDCDelta
                       ///
  CalSAC0,             ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  CalSAC1,             ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  CalSACDelta,         ///< \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  CalSACFourier,       ///< Fourier coefficients of CalSAC1
                       ///
  CalITPC0,            ///< 2D average TPC clusters for longer time interval
  CalITPC1,            ///< 1D integrated TPC clusters
                       ///
  CalCorrMap,          ///< Cluster correction map (high IR rate distortions)
  CalCorrMapRef,       ///< Cluster correction reference map (static distortions)
  CalCorrMapMC,        ///< Cluster correction map (high IR rate distortions) for MC
  CalCorrDerivMapMC,   ///< Cluster correction reference map (static distortions) for MC
                       ///
  CalCorrDerivMap,     ///< Cluster correction map (derivative map)
                       ///
  CalTimeSeries,       ///< integrated DCAs for longer time interval
  CalScaler,           ///< Scaler from IDCs or combined estimator
                       ///
  CorrMapParam,        ///< parameters for CorrectionMapsLoader configuration
                       ///
  DistortionMapMC,     ///< full distortions (static + IR dependant) for MC used in the digitizer
  DistortionMapDerivMC ///< derivative distortions for MC used in the digitizer for scaling
};

/// Storage name in CCDB for each calibration and parameter type
const std::unordered_map<CDBType, const std::string> CDBTypeMap{
  {CDBType::CalPedestal, "TPC/Calib/Pedestal"},
  {CDBType::CalNoise, "TPC/Calib/Noise"},
  {CDBType::CalPedestalNoise, "TPC/Calib/PedestalNoise"},
  {CDBType::CalPulser, "TPC/Calib/Pulser"},
  {CDBType::CalCE, "TPC/Calib/CE"},
  {CDBType::CalPadGainFull, "TPC/Calib/PadGainFull"},
  {CDBType::CalPadGainResidual, "TPC/Calib/PadGainResidual"},
  {CDBType::CalLaserTracks, "TPC/Calib/LaserTracks"},
  {CDBType::CalTimeGain, "TPC/Calib/TimeGain"},
  {CDBType::CalGas, "TPC/Calib/Gas"},
  {CDBType::CalTemperature, "TPC/Calib/Temperature"},
  {CDBType::CalHV, "TPC/Calib/HV"},
  {CDBType::CalTopologyGain, "TPC/Calib/TopologyGainPiecewise"},
  {CDBType::CalVDriftTgl, "TPC/Calib/VDriftTgl"},
  //
  {CDBType::ConfigFEEPad, "TPC/Config/FEEPad"},
  {CDBType::ConfigFEE, "TPC/Config/FEE"},
  {CDBType::ConfigRunInfo, "TPC/Config/RunInfo"},
  //
  {CDBType::ParDetector, "TPC/Parameter/Detector"},
  {CDBType::ParElectronics, "TPC/Parameter/Electronics"},
  {CDBType::ParGas, "TPC/Parameter/Gas"},
  {CDBType::ParGEM, "TPC/Parameter/GEM"},
  // IDCs
  {CDBType::CalIDC0A, "TPC/Calib/IDC_0_A"},
  {CDBType::CalIDC0C, "TPC/Calib/IDC_0_C"},
  {CDBType::CalIDC1A, "TPC/Calib/IDC_1_A"},
  {CDBType::CalIDC1C, "TPC/Calib/IDC_1_C"},
  {CDBType::CalIDCDeltaA, "TPC/Calib/IDC_DELTA_A"},
  {CDBType::CalIDCDeltaC, "TPC/Calib/IDC_DELTA_C"},
  {CDBType::CalIDCFourierA, "TPC/Calib/IDC_FOURIER_A"},
  {CDBType::CalIDCFourierC, "TPC/Calib/IDC_FOURIER_C"},
  {CDBType::CalIDCPadStatusMapA, "TPC/Calib/IDC_PadStatusMap_A"},
  {CDBType::CalIDCPadStatusMapC, "TPC/Calib/IDC_PadStatusMap_C"},
  {CDBType::CalIDCGroupingParA, "TPC/Calib/IDC_GROUPINGPAR_A"},
  {CDBType::CalIDCGroupingParC, "TPC/Calib/IDC_GROUPINGPAR_C"},
  // SACs
  {CDBType::CalSAC0, "TPC/Calib/SAC_0"},
  {CDBType::CalSAC1, "TPC/Calib/SAC_1"},
  {CDBType::CalSACDelta, "TPC/Calib/SAC_DELTA"},
  {CDBType::CalSACFourier, "TPC/Calib/SAC_FOURIER"},
  // ITPCCs
  {CDBType::CalITPC0, "TPC/Calib/ITPCC_0"},
  {CDBType::CalITPC1, "TPC/Calib/ITPCC_1"},
  // correction maps
  {CDBType::CalCorrMap, "TPC/Calib/CorrectionMapV2"},
  {CDBType::CalCorrMapRef, "TPC/Calib/CorrectionMapRefV2"},
  // correction maps for MC
  {CDBType::CalCorrMapMC, "TPC/Calib/CorrectionMapMCV2"},
  {CDBType::CalCorrDerivMapMC, "TPC/Calib/CorrectionMapDerivativeMCV2"},
  // derivative map correction
  {CDBType::CalCorrDerivMap, "TPC/Calib/CorrectionMapDerivativeV2"},
  // time series
  {CDBType::CalTimeSeries, "TPC/Calib/TimeSeries"},
  {CDBType::CalScaler, "TPC/Calib/Scaler"},
  // correction maps loader params
  {CDBType::CorrMapParam, "TPC/Calib/CorrMapParam"},
  // distortion maps
  {CDBType::DistortionMapMC, "TPC/Calib/DistortionMapMC"},
  {CDBType::DistortionMapDerivMC, "TPC/Calib/DistortionMapDerivativeMC"},
};

} // namespace o2::tpc
#endif
