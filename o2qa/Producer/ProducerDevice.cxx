#include <FairMQLogger.h>
#include <TMessage.h>
#include <string>
#include <chrono>
#include <thread>
#include <FairMQTransportFactoryZMQ.h>

#include "ProducerDevice.h"
#include "HistogramProducer.h"
#include "TreeProducer.h"

using namespace std;

ProducerDevice::ProducerDevice(string producerId, string histogramNamePrefix, string histogramTitle, float xLow, float xUp, int numIoThreads) 
{
    this->SetTransport(new FairMQTransportFactoryZMQ);
    this->SetProperty(ProducerDevice::Id, producerId);
    this->SetProperty(ProducerDevice::NumIoThreads, numIoThreads);
    mProducer = make_shared<HistogramProducer>(histogramNamePrefix, histogramTitle, xLow, xUp);
    //mProducer = make_shared<TreeProducer>(histogramId);
}

void freeTMessage(void* data, void* hint)
{
    delete static_cast<TMessage*>(hint);
}

void ProducerDevice::Run()
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

void ProducerDevice::establishChannel(std::string type, std::string method, std::string address, std::string channelName)
{
    FairMQChannel requestChannel(type, method, address);
    requestChannel.UpdateSndBufSize(10000);
    requestChannel.UpdateRcvBufSize(10000);
    requestChannel.UpdateRateLogging(1);
    fChannels[channelName].push_back(requestChannel);
}

void ProducerDevice::executeRunLoop()
{
    ChangeState("INIT_DEVICE");
    WaitForEndOfState("INIT_DEVICE");

    ChangeState("INIT_TASK");
    WaitForEndOfState("INIT_TASK");

    ChangeState("RUN");
    InteractiveStateLoop();
}
