#ifndef LINUX_HELPERS_H
#define LINUX_HELPERS_H

#include <termios.h>
#include <unistd.h>

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
   struct termios term, oterm;
   int fd = 0;
   int c = 0;
   tcgetattr(fd, &oterm);
   term = oterm;
   term.c_lflag = term.c_lflag & (!ICANON);
   term.c_cc[VMIN] = 0;
   term.c_cc[VTIME] = 1;
   tcsetattr(fd, TCSANOW, &term);
   c = getchar();
   tcsetattr(fd, TCSANOW, &oterm);
   if (c != -1)
   ungetc(c, stdin);
   return ((c != -1) ? 1 : 0);
}

static void inline Sleep(int msecs)
{
	usleep(msecs * 1000);
}

#endif
