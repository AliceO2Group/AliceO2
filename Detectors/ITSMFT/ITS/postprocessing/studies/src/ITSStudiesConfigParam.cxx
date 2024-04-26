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

#include "ITSStudies/ITSStudiesConfigParam.h"

namespace o2
{
namespace its
{
namespace study
{
static auto& sAvgClusSizeParamITS = o2::its::study::ITSAvgClusSizeParamConfig::Instance();
static auto& sPIDStudyParamITS = o2::its::study::PIDStudyParamConfig::Instance();
static auto& sCheckTracksParamsITS = o2::its::study::ITSCheckTracksParamConfig::Instance();
static auto& sImpactParameterParamsITS = o2::its::study::ITSImpactParameterParamConfig::Instance();
static auto& sAnomalyStudy = o2::its::study::AnomalyStudyParamConfig::Instance();

O2ParamImpl(o2::its::study::ITSAvgClusSizeParamConfig);
O2ParamImpl(o2::its::study::PIDStudyParamConfig);
O2ParamImpl(o2::its::study::ITSCheckTracksParamConfig);
O2ParamImpl(o2::its::study::ITSImpactParameterParamConfig);
O2ParamImpl(o2::its::study::AnomalyStudyParamConfig);

} // namespace study
} // namespace its
} // namespace o2
