# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_test(roflookuptables
            SOURCES testROFLookupTables.cxx
            COMPONENT_NAME itsmft-tracking
            LABELS "itsmft;tracking"
            PUBLIC_LINK_LIBRARIES O2::ITSMFTTracking)

o2_add_test(slabbumpallocator
            SOURCES testSlabBumpAllocator.cxx
            COMPONENT_NAME itsmft-tracking
            LABELS "itsmft;tracking"
            PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                  O2::ITSMFTTracking
                                  TBB::tbb)

o2_add_test(boundedmemoryresource
            SOURCES testBoundedMemoryResource.cxx
            COMPONENT_NAME itsmft-tracking
            LABELS "itsmft;tracking"
            PUBLIC_LINK_LIBRARIES O2::ITSMFTTracking)

function(o2_add_common_tracking_test name source)
  o2_add_test(${name}
              SOURCES ${source}
              COMPONENT_NAME itsmft-tracking
              LABELS "itsmft;tracking"
              PUBLIC_LINK_LIBRARIES O2::ITSMFTTracking O2::ITStracking O2::MFTTracking)
endfunction()

o2_add_common_tracking_test(detectorlayout testDetectorLayout.cxx)
o2_add_common_tracking_test(traversal-topology testTraversalTopology.cxx)
o2_add_common_tracking_test(mft-normalized-refit testMFTNormalizedRefit.cxx)
o2_add_common_tracking_test(tracklet-finding testTrackletFinding.cxx)
o2_add_common_tracking_test(cell-finding testCellFinding.cxx)
o2_add_common_tracking_test(triplet-fitting testTripletFitting.cxx)
o2_add_common_tracking_test(its-mft-surfacespec-projection testITSMFTSurfaceSpecProjection.cxx)
o2_add_common_tracking_test(generictrack testGenericTrack.cxx)
o2_add_common_tracking_test(propagator testPropagator.cxx)
o2_add_common_tracking_test(material-physics testMaterialPhysics.cxx)
o2_add_common_tracking_test(covariance-sanitization testCovarianceSanitization.cxx)
o2_add_common_tracking_test(surfacetiming testSurfaceTiming.cxx)
o2_add_common_tracking_test(multisourceloading testMultiSourceLoading.cxx)
o2_add_common_tracking_test(timeframe-load-failure testTimeFrameLoadFailure.cxx)
o2_add_common_tracking_test(timeframe-lifecycle testTimeFrameLifecycle.cxx)
o2_add_common_tracking_test(tracker-failure-contract testTrackerFailureContract.cxx)
o2_add_common_tracking_test(computelayercells-orchestration testComputeLayerCellsOrchestration.cxx)
o2_add_common_tracking_test(computelayertracklets-orchestration testComputeLayerTrackletsOrchestration.cxx)
o2_add_common_tracking_test(its-common-ca-tracking-mode-configuration testITSCommonCATrackingModeConfiguration.cxx)
o2_add_test(combined-tracking-composition
            SOURCES testCombinedTrackingComposition.cxx
            COMPONENT_NAME itsmft-tracking
            LABELS "itsmft;tracking"
            PUBLIC_LINK_LIBRARIES O2::ITSMFTTracking O2::ITStracking O2::MFTTracking O2::ITSCAWorkflow)

o2_add_common_tracking_test(mft-ca-tracking-configuration testMFTCATrackingConfiguration.cxx)

o2_add_common_tracking_test(workflow-session testWorkflowSession.cxx)
