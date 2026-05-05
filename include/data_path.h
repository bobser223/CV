//
// Created by Volodymyr Avvakumov on 02.05.2026.
//

#ifndef CODE_DATA_PATH_H
#define CODE_DATA_PATH_H
#include <filesystem>
#include <string>


static const std::string PATH_TO_DATA =
    (std::filesystem::path(__FILE__).parent_path() / "../data/").lexically_normal().string()
    + "/";

#endif //CODE_DATA_PATH_H
