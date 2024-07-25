/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #include "common/test_header.hpp"
#include </u/ewz9kg/cylon_nccl_dev/cpp/src/ncclcylon/utils/util.hpp>
#include </u/ewz9kg/cylon_nccl_dev/cpp/src/ncclcylon/utils/construct.hpp>
#include <iostream>
#include <cylon/ctx/cylon_context.hpp>

using namespace cylon;
//using namespace gcylon;

int main () {
    cylon::Status status;
    const int COLS = 4;
    const int ROWS = 10;
    const int64_t START = 100;
    const int STEP = 5;
    const bool CONT = true;

    std::unique_ptr<cudf::table> tbl = constructTable(COLS, ROWS, START, STEP, CONT);
    auto tv = tbl->view();

    
    if (tv.num_columns() == COLS && tv.num_rows() == ROWS) {
        std::cout << "Columns and rows are correct!" << std::endl;
    }

    int64_t value = START;
    for (int j = 0; j < COLS; j++) {
        int64_t *col = gcylon::getColumnPart<int64_t>(tv.column(j), 0, ROWS);
        if (!CONT)
            value = START;

        for (int i = 0; i < ROWS; i++) {
            if (col[i] != value) {
                std::cout << "Column value is not correct" << std::endl;
            }
            value += STEP;
        }
    }

    std::cout << "Column values are correct" << std::endl;

    std::unique_ptr<cudf::table> tbl2 = constructRandomDataTable(COLS, ROWS);
    auto tv2 = tbl->view();

    if (tv2.num_columns() == COLS && tv2.num_rows() == ROWS){
        std::cout << "Columns and rows are correct for random table!" << std::endl;
    }
    
    return 0;
}

