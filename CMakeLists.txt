cmake_minimum_required (VERSION 2.6)
project (STAT) 

FIND_PACKAGE( OpenCV REQUIRED )

add_executable (stat Main.cpp Statistics.cpp ImageProcessor.cpp Texture.cpp) 

TARGET_LINK_LIBRARIES( stat ${OpenCV_LIBS} )

