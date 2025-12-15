idf_build_get_property(target IDF_TARGET)


set(src_dirs    ./src
                )

set(include_dirs ./include
                )

set(requires        esp_dl
                    spi_flash)

idf_component_register(SRCS ${srcs} SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})