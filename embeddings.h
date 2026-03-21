#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include "tensor.h"



Tensor load_embeddings(const std::string &filename, int vocab_size, int d) {
    Tensor E(vocab_size, d);
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open " << filename << "\n";
        return E;
    }

    std::string line;
    int row = 0;

    // Skip header row
    if (!std::getline(file, line)) {
        std::cerr << "CSV is empty!\n";
        return E;
    }

    while (std::getline(file, line) && row < vocab_size) {
        if (line.empty()) continue;

        // Remove \r for Windows CSVs
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::stringstream ss(line);
        std::string val;

        // First column is the token name — skip it
        if (!std::getline(ss, val, ',')) {
            std::cerr << "Empty line at row " << row << "\n";
            continue;
        }

        int col = 0;
        while (std::getline(ss, val, ',') && col < d) {
            // Trim spaces
            val.erase(0, val.find_first_not_of(" \t\r\n"));
            val.erase(val.find_last_not_of(" \t\r\n") + 1);

            std::cout << "Row " << row << " Col " << col << " val='" << val << "'\n";

            try {
                E(row, col) = std::stof(val);
            } catch (const std::exception &e) {
                std::cerr << "Parse error row=" << row
                          << " col=" << col
                          << " val='" << val << "' -- setting 0.0\n";
                E(row, col) = 0.0f;
            }
            col++;
        }

        if (col < d) {
            std::cerr << "Warning: row " << row << " has only " << col
                      << " values, expected " << d << "\n";
        }

        row++;
    }

    if (row < vocab_size) {
        std::cerr << "Warning: loaded only " << row
                  << " rows, expected " << vocab_size << "\n";
    }

    return E;
}