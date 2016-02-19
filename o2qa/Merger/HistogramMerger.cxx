/**
 * HistogramMerger.cxx
 *
 * @since 2014-10-10
 * @author Patryk Lesiak
 */

#include <FairMQLogger.h>
#include <FairMQTransportFactoryZMQ.h>
#include <chrono>
#include <thread>

#include <TClass.h>

#include "HistogramMerger.h"
#include "HistogramTMessage.h"

using namespace std;

void freeTMessage_sec(void* data, void* hint)
{
    delete static_cast<TMessage*>(hint);
}

HistogramMerger::HistogramMerger(std::string mergerId, int numIoThreads)
{
    this->SetTransport(new FairMQTransportFactoryZMQ);
    this->SetProperty(HistogramMerger::Id, mergerId);
    this->SetProperty(HistogramMerger::NumIoThreads, numIoThreads);
}

void HistogramMerger::CustomCleanup(void* data, void* hint)
{
    delete (string*)hint;
}

void HistogramMerger::establishChannel(std::string type,
                                       std::string method,
                                       std::string address,
                                       std::string channelName)
{
    FairMQChannel requestChannel(type, method, address);
    requestChannel.UpdateSndBufSize(1000);
    requestChannel.UpdateRcvBufSize(1000);
    requestChannel.UpdateRateLogging(1);
    fChannels[channelName].push_back(requestChannel);
}

void HistogramMerger::executeRunLoop()
{
    ChangeState("INIT_DEVICE");
    WaitForEndOfState("INIT_DEVICE");

    ChangeState("INIT_TASK");
    WaitForEndOfState("INIT_TASK");

    ChangeState("RUN");
    InteractiveStateLoop();
}

void HistogramMerger::Run()
{
    const int producerChannel = 0;
    const int controllerChannel = 2;
    unique_ptr<FairMQPoller> poller(fTransportFactory->CreatePoller(fChannels["data"]));

    while (GetCurrentState() == RUNNING) {
        poller->Poll(100);

        if (poller->CheckInput(producerChannel)) {
            LOG(INFO) << "Received histogram from Producer";
            handleReceivedHistograms();
        }

        if (poller->CheckInput(controllerChannel)) {
            LOG(INFO) << "Received controll message";
            handleSystemCommunicationWithController();
        }
    }
}

void HistogramMerger::handleReceivedHistograms()
{
    this_thread::sleep_for(chrono::milliseconds(1000));

    TH1F* receivedHistogram = receiveHistogramFromProducer();

    if (receivedHistogram != nullptr) {
        TCollection* currentHistogramsList = addReceivedObjectToMapByName(receivedHistogram);
        TMessage * viewerMessage = createTMessageForViewer(receivedHistogram, currentHistogramsList);
        unique_ptr<FairMQMessage> viewerReply(fTransportFactory->CreateMessage());

        sendHistogramToViewer(viewerMessage, move(viewerReply));
        sendReplyToProducer(new string("MERGER_OK"));
    }
}

void HistogramMerger::handleSystemCommunicationWithController()
{
    this_thread::sleep_for(chrono::milliseconds(1000));

    unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage());
    fChannels["data"].at(2).Receive(request);
    string* text = new string(GetProperty(HistogramMerger::Id, "default_id") + "_ALIVE");

    FairMQMessage* reply = fTransportFactory->CreateMessage(const_cast<char*>(text->c_str()),
                                                            text->length(),
                                                            CustomCleanup,
                                                            text);
    fChannels["data"].at(2).Send(reply);
}

TCollection* HistogramMerger::addReceivedObjectToMapByName(TObject* receivedObject)
{
    auto foundList = mHistogramIdTohistogramMap.find(receivedObject->GetName());

    if (foundList != mHistogramIdTohistogramMap.end()) {
        foundList->second->Add(receivedObject);
        return foundList->second.get();
    }
    else {   
        auto newItemIterator = mHistogramIdTohistogramMap.insert(make_pair(receivedObject->GetName(),
                                                                           make_shared<TList>()));
        newItemIterator.first->second->SetOwner();
        newItemIterator.first->second->Add(receivedObject);
        return newItemIterator.first->second.get();
    }
}

TMessage* HistogramMerger::createTMessageForViewer(TH1F* receivedHistogram, TCollection* histogramsList)
{
    TMessage* viewerMessage = new TMessage(kMESS_OBJECT);
    shared_ptr<TObject> mergedHistogram(mergeObjectWithGivenCollection(receivedHistogram, histogramsList));
    viewerMessage->WriteObject(mergedHistogram.get());
    return viewerMessage;
}

TObject* HistogramMerger::mergeObjectWithGivenCollection(TObject* object, TCollection* mergeList) 
{
    TObject* mergedObject = object->Clone(object->GetName());

    if (!mergedObject->IsA()->GetMethodWithPrototype("Merge", "TCollection*"))
    {
        LOG(ERROR) << "Object does not implement a merge function!";
        return nullptr;
    }
    Int_t error = 0;
    TString listHargs;
    listHargs.Form("((TCollection*)0x%lx)", (ULong_t) mergeList);

    mergedObject->Execute("Merge", listHargs.Data(), &error);
    if (error)
    {
        LOG(ERROR) << "Error " << error << "running merge!";
        return nullptr;
    }

    return mergedObject;
}

TH1F* HistogramMerger::receiveHistogramFromProducer()
{
    TH1F* receivedHistogram;
    unique_ptr<FairMQMessage> request(fTransportFactory->CreateMessage());
    fChannels["data"].at(0).Receive(request);

    if (request->GetSize() != 0) {
        LOG(INFO) << "Received histogram from histogram producer";
        HistogramTMessage tm(request->GetData(), request->GetSize());
        receivedHistogram = static_cast<TH1F*>(tm.ReadObject(tm.GetClass()));
        LOG(INFO) << "Received histogram name: " << receivedHistogram->GetName();
    }
    else {
        LOG(ERROR) << "Received empty message from producer, skipping RUN procedure";
        receivedHistogram = nullptr;
    }

    return receivedHistogram;
}

void HistogramMerger::sendHistogramToViewer(TMessage* viewerMessage, unique_ptr<FairMQMessage> viewerReply)
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

void HistogramMerger::sendReplyToProducer(std::string* message)
{
    LOG(INFO) << "Sending reply to producer.";
    unique_ptr<FairMQMessage> reply(fTransportFactory->CreateMessage(const_cast<char*>(message->c_str()), 
                                                                     message->length(), 
                                                                     CustomCleanup, 
                                                                     message));
    fChannels["data"].at(0).Send(reply);
}

HistogramMerger::~HistogramMerger()
{
    for (auto const& entry : mHistogramIdTohistogramMap) { 
        entry.second->Delete();
    } 
}
