cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(PainlessMG LANGUAGES CXX CUDA)

add_executable(soft CUDA_Projective_Armadillo/kernel.cu)

target_link_directories(soft PUBLIC 
external_lib/freeglut/lib/x64
external_lib/glew/lib
)

target_link_libraries(soft PUBLIC 
cusparse
cublas
cusolver
freeglut
glew32)

target_include_directories(soft PUBLIC 
lib
external_lib  
external_lib/eigen
external_lib/freeglut/include
external_lib/glew/include)
# set_target_properties(soft PROPERTIES CUDA_SEPARABLE_COMPILATION ON
# CUDA_ARCHITECTURES "60;70;75;80")
set_target_properties(soft PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(soft 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_LIST_DIR}/CUDA_Projective_Armadillo"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_LIST_DIR}/CUDA_Projective_Armadillo"
)