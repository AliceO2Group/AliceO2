// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RecoWorkflow.cxx


#include "DetectorsBase/Propagator.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DeviceSpec.h"
#include "SimReaderSpec.h"
#include "CollisionTimePrinter.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Framework/runDataProcessing.h"


#include "ITSDIGIRECOWorkflow/DIGIRECOWorkFlow.h"
//#include "ITSDIGIRECOWorkflow/HisAnalyzerSpec.h"

#include "ITSDIGIRECOWorkflow/ClustererSpec.h"
#include "ITSDIGIRECOWorkflow/TrackerSpec.h"
#include "ITSDIGIRECOWorkflow/CookedTrackerSpec.h"
#include "ITSDIGIRECOWorkflow/TrackWriterSpec.h"
#include "ITSMFTDigitizerSpec.h"
#include "ITSMFTDigitWriterSpec.h"

namespace o2
{
	namespace ITS
	{

		namespace DIGIRECOWorkFlow
		{

			framework::WorkflowSpec getWorkflow()
			{
				framework::WorkflowSpec specs;
				int fanoutsize = 0;
				std::vector<o2::detectors::DetID> detList;
				detList.emplace_back(o2::detectors::DetID::ITS);
				// connect the ITS digitization
				specs.emplace_back(o2::ITSMFT::getITSDigitizerSpec(fanoutsize++));
				//  specs.emplace_back(o2::ITS::getDigitReaderSpec());
				specs.emplace_back(o2::ITS::getClustererSpec());
				//  specs.emplace_back(o2::ITS::getClusterWriterSpec());
				//specs.emplace_back(o2::ITS::getTrackerSpec());
				specs.emplace_back(o2::ITS::getCookedTrackerSpec());
				specs.emplace_back(o2::ITS::getTrackWriterSpec());

				return specs;
			}

		} // namespace RecoWorkflow

	} // namespace ITS
} // namespace o2
