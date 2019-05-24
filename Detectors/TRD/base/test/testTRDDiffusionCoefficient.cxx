// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTRDDiffusionCoefficient.cxx
/// \brief This task tests the Diffusion Coefficient module of the TRD digitization
/// \author Jorge Lopez, Heidelberg, jlopez@uni-heidelberg

#define BOOST_TEST_MODULE Test TRD DiffusionCoefficient
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TRDBase/TRDCommonParam.h"

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

  auto commonParam = TRDCommonParam::Instance();
  float dl = 0;
  float dt = 0;
  float vd = 1.48;
  commonParam->GetDiffCoeff(dl, dt, vd);
  // check whether the values match the expected AliRoot known output
  BOOST_CHECK_CLOSE(dl, 0.0255211, 0.1);
  BOOST_CHECK_CLOSE(dt, 0.0179734, 0.1);
}

} // namespace trd
} // namespace o2
