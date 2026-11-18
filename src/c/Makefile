# Núcleo de tensores en C (C11, sin dependencias)
#
#   make        compila el test
#   make test   ejecuta los 19 tests
#   make clean  borra binarios

CC      = gcc
CFLAGS  = -O2 -std=c11 -Wall -Wextra
LDLIBS  ?= -lm

BIN = tensor_test

all: $(BIN)

$(BIN): tensor_test.c tensor.c
	$(CC) $(CFLAGS) -o $@ tensor_test.c $(LDLIBS)

test: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN) *.o

.PHONY: all test clean
