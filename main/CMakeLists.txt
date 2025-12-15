# main/CMakeLists.txt

file(GLOB WAV_SOURCES "esp_wav/*.cpp")
file(GLOB MY_MFCC_SOURCES "esp_mfcc/*.c")

set(requires    esp-dl)

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

idf_component_register(SRCS
    "hello_world_main.cpp"
    ${WAV_SOURCES}
    ${MY_MFCC_SOURCES}
    INCLUDE_DIRS
        "."
        "esp_mfcc"
        "esp_wav"
        "../managed_components/espressif__dl_fft"
    PRIV_REQUIRES
        spi_flash
        spiffs
        esp-dl
        esp_adc
)

if (IDF_TARGET STREQUAL "esp32s3")
    set(embed_files "/home/coder/esp/projects/HelloAi/model/esp32s3/xiaoa.espdl")
elseif (IDF_TARGET STREQUAL "esp32p4")
    set(embed_files "/home/coder/esp/projects/HelloAi/model/esp32p4/xiaoa_noise.espdl")
endif()
target_add_aligned_binary_data(${COMPONENT_LIB} ${embed_files} BINARY)
#esptool_py_flash_to_partition(model "/home/coder/esp/projects/HelloAi/model/esp/xiaoa100.espdl")
#esptool_py_flash_to_partition(flash "audio" "${audio_file}")
spiffs_create_partition_image(audio /home/coder/esp/projects/HelloAi/audio_data/flash FLASH_IN_PROJECT)
