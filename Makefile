.PHONY: all
all: k-nn dump

.PHONY: k-nn
k-nn:
	g++ -O4 -o k-nn k-nn.cpp

.PHONY: dump
dump:
	g++ -O4 -o dump dump.cpp  

.PHONY: clean
clean:
	rm -f *.o k-nn dump vgcore.* *.res *.dat a.out
