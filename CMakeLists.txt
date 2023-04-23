cmake_minimum_required(VERSION 3.14)
project(Neural_Network
	HOMEPAGE_URL "https://github.com/victorlaurentiu/Neural_Network"
	LANGUAGES CXX
)

set(TARGET_NAME Neural_Network)

file(GLOB ALL_SRC_FILES src/*)

add_executable(${TARGET_NAME}
	${ALL_SRC_FILES}
)
target_include_directories(${TARGET_NAME} PRIVATE src)
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

if (MSVC)
	target_compile_options(${TARGET_NAME} PRIVATE /WX /W4)
else ()
	target_compile_options(${TARGET_NAME} PRIVATE -Werror -Wall)
endif ()
