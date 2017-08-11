#include <functional>

namespace o2 {
namespace framework {

void *initGUI(const char *) {}
bool pollGUI(void *context, std::function<void(void)> guiCallback) {
  return true;
}
void disposeGUI() {
}

} // namespace framework
} // namespace o2
