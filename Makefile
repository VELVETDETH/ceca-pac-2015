
SRC=./src
INCLUDE=./include
BUILD=./build

CC=icc
CFLAGS=-O3
LDFLAGS=-lrt -fopenmp
CXX=icpc
CXXFLAGS=-O3

all:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $(SRC)/proj.cpp 	-I$(INCLUDE) -o $(BUILD)/proj.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $(SRC)/utility.cpp	-I$(INCLUDE) -o $(BUILD)/utility.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $(SRC)/main.cpp 	-I$(INCLUDE) -o $(BUILD)/main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(BUILD)/proj.o $(BUILD)/utility.o $(BUILD)/main.o -o $(BUILD)/msbeam -L$(BUILD)
