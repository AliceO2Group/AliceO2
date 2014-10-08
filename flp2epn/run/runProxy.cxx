/**
 * runProxy.cxx
 *
 * @since 2013-10-07
 * @author A. Rybalchenko
 */

#include <iostream>
#include <csignal>

#include "FairMQLogger.h"
#include "O2Proxy.h"

#ifdef NANOMSG
  #include "FairMQTransportFactoryNN.h"
#else
  #include "FairMQTransportFactoryZMQ.h"
#endif

using std::cout;
using std::cin;
using std::endl;
using std::stringstream;


O2Proxy proxy;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  proxy.ChangeState(O2Proxy::STOP);
  proxy.ChangeState(O2Proxy::END);

  cout << "Shutdown complete. Bye!" << endl;
  exit(1);
}

static void s_catch_signals (void)
{
  struct sigaction action;
  action.sa_handler = s_signal_handler;
  action.sa_flags = 0;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, NULL);
  sigaction(SIGTERM, &action, NULL);
}

int main(int argc, char** argv)
{
  if ( argc != 11 ) {
    cout << "Usage: testProxy \tID numIoTreads\n"
              << "\t\tinputSocketType inputRcvBufSize inputMethod inputAddress\n"
              << "\t\toutputSocketType outputSndBufSize outputMethod outputAddress\n" << endl;
    return 1;
  }

  s_catch_signals();

  LOG(INFO) << "PID: " << getpid();

#ifdef NANOMSG
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
#endif

  proxy.SetTransport(transportFactory);

  int i = 1;

  proxy.SetProperty(O2Proxy::Id, argv[i]);
  ++i;

  int numIoThreads;
  stringstream(argv[i]) >> numIoThreads;
  proxy.SetProperty(O2Proxy::NumIoThreads, numIoThreads);
  ++i;

  proxy.SetProperty(O2Proxy::NumInputs, 1);
  proxy.SetProperty(O2Proxy::NumOutputs, 1);


  proxy.ChangeState(O2Proxy::INIT);


  proxy.SetProperty(O2Proxy::InputSocketType, argv[i], 0);
  ++i;
  int inputRcvBufSize;
  stringstream(argv[i]) >> inputRcvBufSize;
  proxy.SetProperty(O2Proxy::InputRcvBufSize, inputRcvBufSize, 0);
  ++i;
  proxy.SetProperty(O2Proxy::InputMethod, argv[i], 0);
  ++i;
  proxy.SetProperty(O2Proxy::InputAddress, argv[i], 0);
  ++i;

  proxy.SetProperty(O2Proxy::OutputSocketType, argv[i], 0);
  ++i;
  int outputSndBufSize;
  stringstream(argv[i]) >> outputSndBufSize;
  proxy.SetProperty(O2Proxy::OutputSndBufSize, outputSndBufSize, 0);
  ++i;
  proxy.SetProperty(O2Proxy::OutputMethod, argv[i], 0);
  ++i;
  proxy.SetProperty(O2Proxy::OutputAddress, argv[i], 0);
  ++i;


  proxy.ChangeState(O2Proxy::SETOUTPUT);
  proxy.ChangeState(O2Proxy::SETINPUT);
  proxy.ChangeState(O2Proxy::RUN);


  // wait until the running thread has finished processing.
  boost::unique_lock<boost::mutex> lock(proxy.fRunningMutex);
  while (!proxy.fRunningFinished)
  {
      proxy.fRunningCondition.wait(lock);
  }

  proxy.ChangeState(O2Proxy::STOP);
  proxy.ChangeState(O2Proxy::END);

  return 0;
}

