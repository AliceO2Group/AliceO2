// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test CCDB TestReadWriteAny
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <CCDB/TObjectWrapper.h>
#include <boost/test/unit_test.hpp>
#include "CCDB/Condition.h"
#include "CCDB/Manager.h"
#include "TestClass.h" // local header

namespace o2
{
namespace ccdb
{
/// \brief Test writing/reading arbitrary (non TObject) classes A to CCDB
// a dictionary for TObjectWrapper<A> must exist (next to the one for A)
BOOST_AUTO_TEST_CASE(ReadWriteTest1)
{
  auto cdb = o2::ccdb::Manager::Instance();
  cdb->setDefaultStorage("local://O2CDB");

  TestClass parameter;
  const double TESTVALUE = 20.;
  parameter.mD = TESTVALUE;

  int run = 1;
  auto id = new o2::ccdb::ConditionId("TestParam/Test/Test", run, run, 1, 0);
  auto md = new o2::ccdb::ConditionMetaData();
  cdb->putObjectAny(&parameter, *id, md);

  auto condread = cdb->getCondition("TestParam/Test/Test", run);

  TestClass* readbackparameter = nullptr;
  condread->getObjectAs(readbackparameter);
  BOOST_CHECK(readbackparameter);
  BOOST_CHECK_CLOSE(readbackparameter->mD, TESTVALUE, 1E-6);
}
} // namespace ccdb
} // namespace o2
