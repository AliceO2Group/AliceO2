#ifndef FRAMEWORK_DEBUGGUI_H
#define FRAMEWORK_DEBUGGUI_H

#include <functional>

namespace o2 {
namespace framework {

void *initGUI(const char *name);
bool pollGUI(void *context, std::function<void(void)> guiCallback);
void disposeGUI();

}
}

#endif // FRAMEWORK_DEBUGGUI_H
