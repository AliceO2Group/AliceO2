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

// The MFT workflow exposes the shared publication policy. The session suite
// exercises loading, recovery, completion and cleanup for both detector layouts;
// these checks retain the public MFT publish/skip decision contract.

#define BOOST_TEST_MODULE MFT CA tracker publication decision
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "ITSMFTTracking/Tracker.h"
#include "MFTWorkflow/CATrackerSpec.h"

using namespace o2::mft;

BOOST_AUTO_TEST_CASE(InactiveTrackerAlwaysPublishesEmptyRegardlessOfResultValue)
{
  BOOST_CHECK(decideCATrackerPublicationAction(false, o2::itsmft::tracking::TrackingOutcome::Success) == CATrackerPublicationAction::PublishInactiveEmpty);
  BOOST_CHECK(decideCATrackerPublicationAction(false, o2::itsmft::tracking::TrackingOutcome::RecoverableDropped) == CATrackerPublicationAction::PublishInactiveEmpty);
  BOOST_CHECK(decideCATrackerPublicationAction(false, o2::itsmft::tracking::TrackingOutcome::Structural) == CATrackerPublicationAction::PublishInactiveEmpty);
}

BOOST_AUTO_TEST_CASE(ActiveTrackerWithRecoverableDropSkipsPublication)
{
  BOOST_CHECK(decideCATrackerPublicationAction(true, o2::itsmft::tracking::TrackingOutcome::RecoverableDropped) == CATrackerPublicationAction::SkipDroppedTimeFrame);
}

BOOST_AUTO_TEST_CASE(ActiveTrackerWithNonDroppedResultPublishes)
{
  BOOST_CHECK(decideCATrackerPublicationAction(true, o2::itsmft::tracking::TrackingOutcome::Success) == CATrackerPublicationAction::PublishActiveResult);
  BOOST_CHECK(decideCATrackerPublicationAction(true, o2::itsmft::tracking::TrackingOutcome::Structural) == CATrackerPublicationAction::PublishActiveResult);
}
