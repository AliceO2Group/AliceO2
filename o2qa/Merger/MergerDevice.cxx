#include <FairMQLogger.h>
#include <FairMQTransportFactoryZMQ.h>
#include <chrono>
#include <thread>

#include "MergerDevice.h"
#include "HistogramTMessage.h"

using namespace std;

void freeTMessage_sec(void* data, void* hint)
{
    delete static_cast<TMessage*>(hint);
}

MergerDevice::MergerDevice(unique_ptr<Merger> merger, std::string mergerId, int numIoThreads) : mMerger(move(merger))
{
    this->SetTransport(new FairMQTransportFactoryZMQ);
    this->SetProperty(MergerDevice::Id, mergerId);
    this->SetProperty(MergerDevice::NumIoThreads, numIoThreads);
}

void MergerDevice::CustomCleanup(void* data, void* hint)
{
    delete (string*)hint;
}

void MergerDevice::establishChannel(std::string type, std::string method, std::string address, std::string channelName)
{
    FairMQChannel requestChannel(type, method, address);
    requestChannel.UpdateSndBufSize(1000);
    requestChannel.UpdateRcvBufSize(1000);
    requestChannel.UpdateRateLogging(1);
    fChannels[channelName].push_back(requestChannel);
}

void MergerDevice::executeRunLoop()
{
    ChangeState("INIT_DEVICE");
    WaitForEndOfState("INIT_DEVICE");

    ChangeState("INIT_TASK");
    WaitForEndOfState("INIT_TASK");

    ChangeState("RUN");
    InteractiveStateLoop();
}

void MergerDevice::Run()
{
    const int producerChannel = 0;
    const int controllerChannel = 2;
    unique_ptr<FairMQPoller> poller(fTransportFactory->CreatePoller(fChannels["data"]));

    while (GetCurrentState() == RUNNING) {
        poller->Poll(100);

        if (poller->CheckInput(producerChannel)) {
            LOG(INFO) << "Received histogram from Producer";
            handleReceivedDataObject();
        }

        if (poller->CheckInput(controllerChannel)) {
            LOG(INFO) << "Received controll message";
            handleSystemCommunicationWithController();
        }
    }
}

void MergerDevice::handleReceivedDataObject()
{
    this_thread::sleep_for(chrono::milliseconds(1000));

    TObject* receivedObject = receiveDataObjectFromProducer();

    if (receivedObject != nullptr) {
        shared_ptr<TObject> mergedHistogram(mMerger->mergeObject(receivedObject));

        TMessage* viewerMessage = createTMessageForViewer(mergedHistogram);
        unique_ptr<FairMQMessage> viewerReply(fTransportFactory->CreateMessage());

        sendMergedObjectToViewer(viewerMessage, move(viewerReply));
        sendReplyToProducer(new string("MERGER_OK"));
    }
}

TMessage* MergerDevice::createTMessageForViewer(shared_ptr<TObject> objectToSend)
{
    TMessage* viewerMessage = new TMessage(kMESS_OBJECT);
    viewerMessage->WriteObject(objectToSend.get());
    return viewerMessage;
}

void MergerDevice::handleSystemCommunicationWithController()
{
    this_thread::sleep_for(chrono::milliseconds(1000));

    unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage());
    fChannels["data"].at(2).Receive(request);
    string* text = new string(GetProperty(MergerDevice::Id, "default_id") + "_ALIVE");

    FairMQMessage* reply = fTransportFactory->CreateMessage(const_cast<char*>(text->c_str()),
                                                            text->length(),
                                                            CustomCleanup,
                                                            text);
    fChannels["data"].at(2).Send(reply);
}

TObject* MergerDevice::receiveDataObjectFromProducer()
{
    TObject* receivedDataObject;
    unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage());
    fChannels["data"].at(0).Receive(request);

    if (request->GetSize() != 0) {
        LOG(INFO) << "Received histogram from histogram producer";
        HistogramTMessage tm(request->GetData(), request->GetSize());
        receivedDataObject = static_cast<TObject*>(tm.ReadObject(tm.GetClass()));
        LOG(INFO) << "Received histogram name: " << receivedDataObject->GetName();
    }
    else {
        LOG(ERROR) << "Received empty message from producer, skipping RUN procedure";
        receivedDataObject = nullptr;
    }

    return receivedDataObject;
}

void MergerDevice::sendMergedObjectToViewer(TMessage* viewerMessage, unique_ptr<FairMQMessage> viewerReply)
{
    unique_ptr<FairMQMessage> viewerRequest(fTransportFactory->CreateMessage(viewerMessage->Buffer(), 
                                                                             viewerMessage->BufferSize(), 
                                                                             freeTMessage_sec, 
                                                                             viewerMessage));
    LOG(INFO) << "Sending new histogram to viewer";

    fChannels["data"].at(1).Send(viewerRequest);
    fChannels["data"].at(1).Receive(viewerReply);
    
    if (viewerReply->GetSize() != 0) {
        LOG(INFO) << "Received reply from VIEWER: \"" << string(static_cast<char*>(viewerReply->GetData()), 
                                                                                    viewerReply->GetSize()) << "\"";
    }    
    else {
        LOG(ERROR) << "Received empty message from viewer, skipping RUN procedure";
    }
}

void MergerDevice::sendReplyToProducer(std::string* message)
{
    LOG(INFO) << "Sending reply to producer.";
    unique_ptr<FairMQMessage> reply(fTransportFactory->CreateMessage(const_cast<char*>(message->c_str()), 
                                                                     message->length(), 
                                                                     CustomCleanup, 
                                                                     message));
    fChannels["data"].at(0).Send(reply);
}


