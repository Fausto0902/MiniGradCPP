cmake_minimum_required(VERSION 3.12)
project(MiniGradCPP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /WX /permissive-)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_executable(MiniGrad
    main.cpp
    Serializer.cpp
)

target_include_directories(MiniGrad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
