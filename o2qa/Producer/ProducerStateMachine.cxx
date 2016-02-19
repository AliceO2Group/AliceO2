#include <FairMQLogger.h>
#include <TMessage.h>
#include <string>
#include <chrono>
#include <thread>
#include <FairMQTransportFactoryZMQ.h>

#include "ProducerStateMachine.h"
#include "HistogramProducer.h"

using namespace std;

ProducerStateMachine::ProducerStateMachine(string producerId, string histogramId, float xLow, float xUp, int numIoThreads) 
{
    this->SetTransport(new FairMQTransportFactoryZMQ);
    this->SetProperty(ProducerStateMachine::Id, producerId);
    this->SetProperty(ProducerStateMachine::NumIoThreads, numIoThreads);
    mProducer = make_shared<HistogramProducer>(histogramId, xLow, xUp);
}

void freeTMessage(void* data, void* hint)
{
    delete static_cast<TMessage*>(hint);
}

void ProducerStateMachine::Run()
{
    while (GetCurrentState() == RUNNING) {
        this_thread::sleep_for(chrono::milliseconds(1000));

        TObject* newDataObject = mProducer->produceData();
        TMessage * message = new TMessage(kMESS_OBJECT);
        message->WriteObject(newDataObject);
        delete newDataObject;

        unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage(message->Buffer(), 
                                                                  message->BufferSize(), 
                                                                  freeTMessage, 
                                                                  message));
        unique_ptr<FairMQMessage> reply(fTransportFactory->CreateMessage());

        LOG(INFO) << "Sending new histogram to merger";

        fChannels["data"].at(0).Send(request);
        fChannels["data"].at(0).Receive(reply);

        if (reply->GetSize() != 0) {
            LOG(INFO) << "Received reply from merger: \"" 
                      << string(static_cast<char*>(reply->GetData()), reply->GetSize())
                      << "\"";
        }
        else {
            LOG(ERROR) << "Did not Receive reply from merger";
        }
    }
}

void ProducerStateMachine::establishChannel(std::string type, std::string method, std::string address, std::string channelName)
{
    FairMQChannel requestChannel(type, method, address);
    requestChannel.UpdateSndBufSize(10000);
    requestChannel.UpdateRcvBufSize(10000);
    requestChannel.UpdateRateLogging(1);
    fChannels[channelName].push_back(requestChannel);
}

void ProducerStateMachine::executeRunLoop()
{
    ChangeState("INIT_DEVICE");
    WaitForEndOfState("INIT_DEVICE");

    ChangeState("INIT_TASK");
    WaitForEndOfState("INIT_TASK");

    ChangeState("RUN");
    InteractiveStateLoop();
}

void ProducerStateMachine::CustomCleanup(void* data, void* hint)
{
    delete (string*)hint;
}

