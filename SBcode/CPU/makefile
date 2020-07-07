SRCS= sb.c
INCLUDES= 
#DEFINES= -DNCPU=$(N)
DEFINES= -DNCPU=2
PRECIOUS= $(SRCS) $(INCLUDES) Makefile

OBJS= sb.o 

TARGET= sb$(N).x

LOPTS= -fopenmp -lm
OPTFLAGS= -O3 
OKFLAGS= -Wall  $(OPTFLAGS) $(LOPTS)
DEBUGFLAGS=  -Wall -g -lm
all: $(TARGET)

clean: objclean
	rm -f $(TARGET)

objclean: crapclean
	rm -f $(OBJS)

crapclean:
	rm -f *~

$(TARGET): $(OBJS)
	gcc $(OBJS) $(OKFLAGS) -o $@

$(OBJS): CFLAGS= $(OKFLAGS) $(DEFINES)

$(OBJS): $(PRECIOUS)


