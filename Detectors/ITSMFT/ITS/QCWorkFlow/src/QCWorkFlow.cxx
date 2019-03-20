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

#include "ITSQCWorkflow/QCWorkFlow.h"
#include "ITSQCWorkflow/HisAnalyzerSpec.h"
#include "/data/zhaozhong/alice/O2/Detectors/ITSMFT/ITS/workflow/include/ITSWorkflow/DigitReaderSpec.h"
#include "/data/zhaozhong/alice/O2/Detectors/ITSMFT/ITS/workflow/src/DigitReaderSpec.cxx"
namespace o2
{
	namespace ITS
	{

		namespace QCWorkFlow
		{

			framework::WorkflowSpec getWorkflow()
			{
				framework::WorkflowSpec specs;
		
				specs.emplace_back(o2::ITS::getDigitReaderSpec());
			        specs.emplace_back(o2::ITS::getHisAnalyzerSpec());

				return specs;
			}

		} // namespace RecoWorkflow

	} // namespace ITS
} // namespace o2
