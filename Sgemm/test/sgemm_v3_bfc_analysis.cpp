#include <iostream>
#include "./include/util.hpp"

int main() {
    // WRITE SMem_B
    int row[32];
    int col[32];
    int OFFSET = 0;
    int SMem_A_Row = 8;
    int SMem_A_Col = 128;
    for(int tid = 0; tid < 32; ++tid) {
        row[tid] = (tid * 4) % SMem_A_Row;
        col[tid] = (tid * 4) / SMem_A_Row;
    }
    int tid = 0, warpId = tid / 32;
    // bank_conflict_label("WRITE: GMem ==> SMem_A(Transpose)",tid, row, col, SMem_A_Row, SMem_A_Col, OFFSET);

    int SMem_B_Row = 8;
    int SMem_B_Col = 128;
    OFFSET = 4;
    for (int tid = 0; tid < 32; ++tid) {
        row[tid] = (tid * 4) / SMem_B_Col;
        col[tid] = (tid * 4) % SMem_B_Col;
    }
    tid = 0, warpId = tid / 32;
    bank_conflict_label("WRITE: GMem ==> SMem_B",tid, row, col, SMem_B_Row, SMem_B_Col, OFFSET);

    OFFSET = 4;
    SMem_B_Row = 8;
    SMem_B_Col = 128;
    for(int tid = 0; tid < 32; ++tid) {
        row[tid] = 0;
        col[tid] = ((tid * 4) / 64) * 4;
    }
    tid = 0, warpId = tid / 32;
    // bank_conflict_label("READ: SMem_A ==> Reg_A",tid, row, col, SMem_A_Row, SMem_A_Col, OFFSET);

    OFFSET = 4;
    for(int tid = 0; tid < 32; ++tid) {
        row[tid] = 0;
        col[tid] = ((tid * 4) % 64);
    }
    tid = 0, warpId = tid / 32;
    // bank_conflict_label("READ: SMem_B ==> Reg_B",tid, row, col, SMem_A_Row, SMem_A_Col, OFFSET);
    return 0;
}