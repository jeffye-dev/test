cmake_minimum_required(VERSION 3.22)
project(flash_attention_min LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(flash_attention_min src/flash_attention_min.cu)
set_target_properties(flash_attention_min PROPERTIES CUDA_ARCHITECTURES 100)
target_compile_options(flash_attention_min PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
