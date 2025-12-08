# main/CMakeLists.txt
file(GLOB COMMON_SOURCES "audio/common/*.cpp")
file(GLOB MFCC_SOURCES "audio/speech_features/*.cpp")
file(GLOB WAV_SOURCES "esp_wav/*.cpp")
file(GLOB MY_MFCC_SOURCES "esp_mfcc/*.c")

idf_component_register(SRCS
    "hello_world_main.cpp"
    "dl_tensor_base.cpp"
    ${COMMON_SOURCES}
    ${MFCC_SOURCES}
    ${WAV_SOURCES}
    ${MY_MFCC_SOURCES}
    INCLUDE_DIRS
        "."
        "audio/common"
        "esp_mfcc"
        "esp_wav"
        "../managed_components/espressif__dl_fft"
    PRIV_REQUIRES
        spi_flash
        spiffs
        esp-dl
        esp_adc
)
set(image_file "/home/coder/esp/projects/HelloAi/model/esp32p4/xiaoa_noise.espdl")

esptool_py_flash_to_partition(flash "model" "${image_file}")
#esptool_py_flash_to_partition(model "/home/coder/esp/projects/HelloAi/model/esp/xiaoa100.espdl")
#esptool_py_flash_to_partition(flash "audio" "${audio_file}")
spiffs_create_partition_image(audio /home/coder/esp/projects/HelloAi/audio_data/flash FLASH_IN_PROJECT)
