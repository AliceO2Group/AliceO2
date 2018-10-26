// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_gl_Debug3DUtils_H_INCLUDED
#define o2_framework_gl_Debug3DUtils_H_INCLUDED

namespace o2
{
namespace framework
{
namespace gl
{

/// Call this in your initialization callback if you need to do 3d stuff.
void init3DContext(void* context);
/// Simple function to draw a triangle. Just to test everything is fine.
void render3D();
/// To be called on exit
void dispose3DContext();

} // namespace gl
} // namespace framework
} // namespace o2

#endif
