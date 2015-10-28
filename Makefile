
SRC=./src
INCLUDE=./include
BUILD=./build

CC=icc
CFLAGS=-O3
OPENCV_LDFLAGS=-lopencv_core -lopencv_highgui
LDFLAGS=-lrt -openmp $(OPENCV_LDFLAGS)
CXX=icpc
CXXFLAGS=-O3

OBJECTS = $(BUILD)/main.o $(BUILD)/proj.o $(BUILD)/utility.o $(BUILD)/image_toolbox.o \
					$(BUILD)/wray.o

MSBEAM_CPU = $(BUILD)/msbeam_cpu.o
MSBEAM_OFFLOAD_CPU = $(BUILD)/msbeam_offload_cpu.o
MSBEAM_MIC = $(BUILD)/msbeam_mic.o
MSBEAM = $(MSBEAM_CPU) $(MSBEAM_OFFLOAD_CPU) $(MSBEAM_MIC)

$(BUILD)/%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $< -I$(INCLUDE) -o $@

$(BUILD)/msbeam: $(OBJECTS) $(MSBEAM)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(MSBEAM) \
	-o $(BUILD)/msbeam -L$(BUILD)

msbeam: $(BUILD)/msbeam 

clean:
	rm -rf $(BUILD)/*