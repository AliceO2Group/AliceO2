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

/// \file testTRDDiffusionCoefficient.cxx
/// \brief This task tests the Diffusion Coefficient module of the TRD digitization
/// \author Jorge Lopez, Universitaet Heidelberg, jlopez@physi.uni-heidelberg.de

#define BOOST_TEST_MODULE Test TRD DiffusionCoefficient
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TRDBase/CommonParam.h"
#include "TRDBase/DiffAndTimeStructEstimator.h"

#include "Field/MagneticField.h"
#include <TGeoGlobalMagField.h>

namespace o2
{
namespace trd
{

/// \brief Test 1 of the GetDiffCoeff function
BOOST_AUTO_TEST_CASE(TRDDiffusionCoefficient_test1)
{
  auto fld = o2::field::MagneticField::createFieldMap();
  TGeoGlobalMagField::Instance()->SetField(fld);
  TGeoGlobalMagField::Instance()->Lock();

  auto commonParam = CommonParam::instance();
  float dl = 0;
  float dt = 0;
  float vd = 1.48;
  commonParam->getDiffCoeff(dl, dt, vd);
  // check whether the values match the expected AliRoot known output
  BOOST_CHECK_CLOSE(dl, 0.0255211, 0.1);
  BOOST_CHECK_CLOSE(dt, 0.0179734, 0.1);
}

/// \brief Test time structure
BOOST_AUTO_TEST_CASE(TRDTimeStructure_test)
{
  auto commonParam = CommonParam::instance();
  DiffusionAndTimeStructEstimator estimator;
  BOOST_CHECK_CLOSE(estimator.TimeStruct(1.48, 1., 0.1), commonParam->timeStruct(1.48, 1., 0.1), 0.001);
  BOOST_CHECK_CLOSE(estimator.TimeStruct(1.1, 1., 0.1), commonParam->timeStruct(1.1, 1., 0.1), 0.001);
  BOOST_CHECK_CLOSE(estimator.TimeStruct(2, 1., 0.1), commonParam->timeStruct(2, 1., 0.1), 0.001);
  BOOST_CHECK_CLOSE(estimator.TimeStruct(4, 1., 0.1), commonParam->timeStruct(4, 1., 0.1), 0.001);
}

/// \brief compare diffusion coeff
BOOST_AUTO_TEST_CASE(TRDDiffusion_test)
{
  auto commonParam = CommonParam::instance();
  DiffusionAndTimeStructEstimator estimator;
  float dl1 = 0.;
  float dl2 = 0.;
  float dt1 = 0.;
  float dt2 = 0.;
  estimator.GetDiffCoeff(dl1, dt1, 1.48);
  commonParam->getDiffCoeff(dl2, dt2, 1.48);
  BOOST_CHECK_CLOSE(dl1, dl2, 0.001);
  BOOST_CHECK_CLOSE(dt1, dt2, 0.001);

  estimator.GetDiffCoeff(dl1, dt1, 1.1);
  commonParam->getDiffCoeff(dl2, dt2, 1.1);
  BOOST_CHECK_CLOSE(dl1, dl2, 0.001);
  BOOST_CHECK_CLOSE(dt1, dt2, 0.001);

  estimator.GetDiffCoeff(dl1, dt1, 2);
  commonParam->getDiffCoeff(dl2, dt2, 2);
  BOOST_CHECK_CLOSE(dl1, dl2, 0.001);
  BOOST_CHECK_CLOSE(dt1, dt2, 0.001);

  estimator.GetDiffCoeff(dl1, dt1, 4);
  commonParam->getDiffCoeff(dl2, dt2, 4);
  BOOST_CHECK_CLOSE(dl1, dl2, 0.001);
  BOOST_CHECK_CLOSE(dt1, dt2, 0.001);
}

} // namespace trd
} // namespace o2
