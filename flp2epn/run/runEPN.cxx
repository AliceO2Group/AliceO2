/**
 * runEPN.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#include <iostream>
#include <csignal>

#include "FairMQLogger.h"
#include "O2EPNex.h"

#ifdef NANOMSG
  #include "FairMQTransportFactoryNN.h"
#else
  #include "FairMQTransportFactoryZMQ.h"
#endif

using std::cout;
using std::cin;
using std::endl;
using std::stringstream;


O2EPNex epn;

static void s_signal_handler (int signal)
{
  cout << endl << "Caught signal " << signal << endl;

  epn.ChangeState(O2EPNex::STOP);
  epn.ChangeState(O2EPNex::END);

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
  if ( argc != 7 ) {
    cout << "Usage: testEPN \tID numIoTreads\n"
              << "\t\tinputSocketType inputRcvBufSize inputMethod inputAddress\n"
              << endl;
    return 1;
  }

  s_catch_signals();

  LOG(INFO) << "PID: " << getpid();

#ifdef NANOMSG
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryNN();
#else
  FairMQTransportFactory* transportFactory = new FairMQTransportFactoryZMQ();
#endif

  epn.SetTransport(transportFactory);

  int i = 1;

  epn.SetProperty(O2EPNex::Id, argv[i]);
  ++i;

  int numIoThreads;
  stringstream(argv[i]) >> numIoThreads;
  epn.SetProperty(O2EPNex::NumIoThreads, numIoThreads);
  ++i;

  epn.SetProperty(O2EPNex::NumInputs, 1);
  epn.SetProperty(O2EPNex::NumOutputs, 0);


  epn.ChangeState(O2EPNex::INIT);


  epn.SetProperty(O2EPNex::InputSocketType, argv[i], 0);
  ++i;
  int inputRcvBufSize;
  stringstream(argv[i]) >> inputRcvBufSize;
  epn.SetProperty(O2EPNex::InputRcvBufSize, inputRcvBufSize, 0);
  ++i;
  epn.SetProperty(O2EPNex::InputMethod, argv[i], 0);
  ++i;
  epn.SetProperty(O2EPNex::InputAddress, argv[i], 0);
  ++i;


  epn.ChangeState(O2EPNex::SETOUTPUT);
  epn.ChangeState(O2EPNex::SETINPUT);
  epn.ChangeState(O2EPNex::RUN);

  // wait until the running thread has finished processing.
  boost::unique_lock<boost::mutex> lock(epn.fRunningMutex);
  while (!epn.fRunningFinished)
  {
      epn.fRunningCondition.wait(lock);
  }

  epn.ChangeState(O2EPNex::STOP);
  epn.ChangeState(O2EPNex::END);

  return 0;
}
