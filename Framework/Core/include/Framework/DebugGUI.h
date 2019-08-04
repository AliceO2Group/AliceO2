// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DEBUGGUI_H
#define FRAMEWORK_DEBUGGUI_H

#include <functional>

namespace o2
{
namespace framework
{

void* initGUI(const char* name);
bool pollGUI(void* context, std::function<void(void)> guiCallback);
void disposeGUI();

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DEBUGGUI_H
