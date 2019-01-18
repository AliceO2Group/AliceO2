#ifndef LINUX_HELPERS_H
#define LINUX_HELPERS_H

#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>

static inline int getch()
{
	static struct termios oldt, newt;
	tcgetattr( STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON|ECHO);
	tcsetattr( STDIN_FILENO, TCSANOW, &newt);
	int retVal = getchar();
	tcsetattr( STDIN_FILENO, TCSANOW, &oldt);
	return(retVal);
}

static inline int kbhit()
{
	termios term;
	tcgetattr(0, &term);
	termios term2 = term;
	term2.c_lflag &= ~ICANON;
	tcsetattr(0, TCSANOW, &term2);
	int byteswaiting;
	ioctl(0, FIONREAD, &byteswaiting);
	tcsetattr(0, TCSANOW, &term);
	return byteswaiting > 0;
}

static void inline Sleep(int msecs)
{
	usleep(msecs * 1000);
}

#endif
