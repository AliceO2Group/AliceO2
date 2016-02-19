/**
 * runProducerStateMachine.cxx
 *
 * @since 2015-09-30
 * @author Patryk Lesiak
 */

#include <csignal>
#include <FairMQLogger.h>
#include <cstdlib>
#include <vector>

#include "ProducerStateMachine.h"

namespace
{

std::vector<ProducerStateMachine*> producerStateMachines;

void signalHandler(int signal)
{
    LOG(INFO) << "Caught signal " << signal;

    for (auto ProducerStateMachine : producerStateMachines) {
        ProducerStateMachine->ChangeState(ProducerStateMachine::END);
    }

    LOG(INFO) << "Shutdown complete.";
}

}

int main(int argc, char** argv)
{
    constexpr int requiredNumberOfProgramParameters{4};

    if (argc != requiredNumberOfProgramParameters) {
        LOG(ERROR) << "Wrong number of program parameters, required three parameters: histogram xLow, xUp and Id";
        return -1;
    }

    ProducerStateMachine ProducerStateMachine("Producer", argv[3], atof(argv[1]), atof(argv[2]), 1);
    producerStateMachines.push_back(&ProducerStateMachine);
    
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    LOG(INFO) << "PID: " << getpid();
    LOG(INFO) << "Producer id: " 
              << producerStateMachines[0]->GetProperty(ProducerStateMachine::Id, "default_id");

    producerStateMachines[0]->establishChannel("req", "connect", "tcp://localhost:5005", "data");

    producerStateMachines[0]->executeRunLoop();

    LOG(INFO) << "END OF runProducerStateMachine";

    return 0;
}
