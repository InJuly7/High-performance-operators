#include <iostream>
#include <iomanip>
#include <map>

#define BANKWORD 4
const int SMem_Row = 16;
const int SMem_Col = 256;
// half 16bits ==> 2B float 32bits ==> 4B
const int Element_size = 2;
// half8 ==> 16B ==> 4bit
const int Chunk_size = 16;
const int HALF_BITS = 1;
const int FLOAT_BITS = 2;
const int INT_BITS = 2;

void print_bankId(std::pair<int,int> Bank[][SMem_Col], int row, int col) {
	// 设置输出宽度为3个字符 如果输出内容不足3位,会用空格填充(默认右对齐)
    std::cout << "layer: " << std::setw(3) << Bank[row][col].first << " bankId: " << std::setw(3) << Bank[row][col].second << std::endl;
}


void Compute_BankId(std::pair<int,int> Bank[][SMem_Col], int row, int col) {
    int addr = row * SMem_Col * Element_size + col * Element_size;
    addr = (addr / BANKWORD) * BANKWORD;
	int layer = addr / (32 * BANKWORD);
    int bankId = (addr % (32 * BANKWORD)) / BANKWORD;
	// layer, bankId
    Bank[row][col] = {layer, bankId};
}

void Compute_Swizzle_BankId(std::pair<int,int> Bank[][SMem_Col], int row, int col, int swizzle_col) {
    int addr = row * SMem_Col * Element_size + swizzle_col * Element_size;
    addr = (addr / BANKWORD) * BANKWORD;
	int layer = addr / (32 * BANKWORD);
    int bankId = (addr % (32 * BANKWORD)) / BANKWORD;
	// layer, bankId
    Bank[row][col] = {layer, bankId};
}

void printBinary(int num) {
    for (int i = 31; i >= 0; i--) {
        std::cout << ((num >> i) & 1);
        if (i % 4 == 0 && i != 0) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;
}

template<unsigned int S, unsigned int B, unsigned int M> 
int Swizzle(int row, int col) {
	int swizzle_col = 0;
	int addr = row * SMem_Col * Element_size + col * Element_size;
	// printBinary(addr);
    int BMask = ((1 << B) - 1) << M;
	// printBinary(BMask);
	int swizzle_addr = ((addr >> S) & BMask) ^ addr;
	// printBinary(swizzle_addr);
	// half, 2B, 1bits
	swizzle_col = (swizzle_addr & ((1 << (B + M)) - 1)) >> HALF_BITS;
	// printBinary(swizzle_col);
	return swizzle_col;
}


int main()
{
	// 每个元素的 layer 与  bankId
    std::pair<int,int> Bank[SMem_Row][SMem_Col];
	// 每个元素 Swizzle 之后 对应的 列
	int Swizzle_arr[SMem_Row][SMem_Col];
	// 存储 Swizzle 之后的 Bank layer 与 bankId
	std::pair<int,int> Swizzle_Bank[SMem_Row][SMem_Col];
	for(int i = 0; i < SMem_Row; i ++)
	{
		for(int j = 0; j < SMem_Col; j ++)
		{
			Compute_BankId(Bank, i, j);
		}
	}

	// for(int i = 0; i < 8; i++) {
	// 	for(int j = 0; j < 8; j++) {
	// 		print_bankId(Bank, i, j);
	// 	}
	// 	std::cout << std::endl;
    // }
    

	for(int i = 0; i < 16; i ++)
	{
		for(int j = 0; j < 256; j ++)
		{
			Swizzle_arr[i][j] = Swizzle<5,3,4>(i, j);
			// std::cout << "Row: " << i << " Col: " << j << " Swizzle_col: " << Swizzle_arr[i][j] << std::endl;
			Compute_Swizzle_BankId(Swizzle_Bank, i, j, Swizzle_arr[i][j]);
		}
		// std::cout << "----------------------"<<  std::endl;
	}

    for (int i = 1; i < 2; i++) {
        for (int j = 0; j < 256; j++) {
            print_bankId(Swizzle_Bank, i, j);
        }
        std::cout << "----------------------"<<  std::endl;
    }

	return 0;
}