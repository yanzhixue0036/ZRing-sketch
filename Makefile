
# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -g


# Include paths
INCLUDES = -I./cnpy -I./include

# Libraries
LIBS = -lz -pthread

# Source files
SRCS = main.cpp cnpy/cnpy.cpp include/MurmurHash3.cpp

# Target binary name
TARGET = main

# Default rule
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRCS) $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)
	rm -f ./tmp/*

# Run rule
run: $(TARGET)
	./$(TARGET) -m QSketchDyn -k 16

.PHONY: all clean run
