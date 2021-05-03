// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDTRAPSIMULATORTRACKLETWRITER_H
#define O2_TRDTRAPSIMULATORTRACKLETWRITER_H

namespace o2
{
namespace framework
{
struct DataProcessorSpec;
}
} // namespace o2

namespace o2
{
namespace trd
{

o2::framework::DataProcessorSpec getTRDTrackletWriterSpec(bool useMC);

} // end namespace trd
} // end namespace o2

#endif // O2_TRDTRAPSIMULATORTRACKLETWRITER_H
