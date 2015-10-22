
SRC=./src
INCLUDE=./include
BUILD=./build

CC=icc
CFLAGS=-O3
LDFLAGS=-lrt -fopenmp
CXX=icpc
CXXFLAGS=-O3

OBJECTS = $(BUILD)/main.o $(BUILD)/proj.o $(BUILD)/utility.o $(BUILD)/image_toolbox.o
MSBEAM_CPU = $(BUILD)/msbeam_cpu.o
MSBEAM_MIC = $(BUILD)/msbeam_mic.o

$(BUILD)/%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $< -I$(INCLUDE) -o $@

cpu: $(OBJECTS) $(MSBEAM_CPU)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(MSBEAM_CPU) \
	-o $(BUILD)/msbeam -L$(BUILD)

mic: $(OBJECTS) $(MSBEAM_MIC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(MSBEAM_MIC) \
	-o $(BUILD)/msbeam -L$(BUILD)
