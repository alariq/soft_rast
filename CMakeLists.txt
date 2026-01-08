cmake_minimum_required(VERSION 3.6)

project(soft_rast)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED ${CMAKE_BUILD_TYPE})
    #set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    set(CMAKE_BUILD_TYPE "Debug")
    message(" -- setting CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
endif()

# set the output directory for built objects.
# This makes sure that the dynamic library goes into the build directory automatically.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

set(THIRD_PARTY "3rdparty/")

# This assumes the SDL source is available in vendored/SDL
add_subdirectory(3rdparty/SDL3-3.2.14 EXCLUDE_FROM_ALL)

find_library(OPENG_LIB NAMES GL)
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# Dear ImGui
set(IMGUI_DIR "${THIRD_PARTY}/imgui/")
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends)

#add_definitions(-DIMGUI_IMPL_OPENGL_ES2)

file(GLOB IMGUI_SRC ${IMGUI_DIR}/*.cpp ${IMGUI_DIR}backends/imgui_impl_opengl3.cpp ${IMGUI_DIR}backends/imgui_impl_sdl3.cpp)
file(GLOB IMGUI_INC ${IMGUI_DIR}/*.h ${IMGUI_DIR}backends/imgui_impl_opengl3.h ${IMGUI_DIR}backends/imgui_impl_sdl3.h)

list(APPEND SRC ${IMGUI_SRC})
list(APPEND INC ${IMGUI_INC})

# main app
file(GLOB APP_SRC *.cpp)
file(GLOB APP_INC *.h)
list(APPEND SRC ${APP_SRC})
list(APPEND INC ${APP_INC})


# -fplugin=externis -fplugin-arg-externis-trace=trace.json")
add_compile_options(-Wall -faligned-new -Werror -Wfatal-errors) # -pedantic
add_compile_options(-Wno-gnu-anonymous-struct)
add_compile_options(-Wno-nested-anon-types)

option(WITH_ASAN "Enable address sanitizer" TRUE)
if(${WITH_ASAN})
    add_compile_options(-fsanitize=address -static-libasan)
    set(ASAN_LIB asan)
endif()
#add_compile_options(-finstrument-functions)
set (CMAKE_CXX_STANDARD 14)


#set(CMAKE_CXX_FLAGS_DEBUG  "-Og -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG  " -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE  "-O2")


find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3f)
find_library(hackrf REQUIRED NAMES hackrf)

add_executable(soft_rast ${SRC} ${INC})
# asan should be linked first
target_link_libraries(soft_rast PRIVATE ${ASAN_LIB} pthread hackrf fftw3f SDL3::SDL3 ${OPENG_LIB} ${GLEW_LIBRARIES} )

install(TARGETS soft_rast DESTINATION "${INSTALL_PREFIX}opt/bin")

