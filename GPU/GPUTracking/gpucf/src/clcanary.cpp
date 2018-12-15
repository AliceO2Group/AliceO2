#include <gpucf/ClCanary.h>
#include <gpucf/ClEnv.h>
#include <gpucf/log.h>


int main() {
    try {
        ClEnv env; 
        ClCanary canary;

        canary.run(env);

        return 0;
    } catch (const Exception &e) {
        log::Error() << "Caught exception: " << e.what();

        return 1;
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
