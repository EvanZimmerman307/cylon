#include <iostream>
#include <fstream>
#include <random>
#include <iomanip> // For std::setprecision
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>


// Function to generate a CSV file with 5 columns of random numbers
void generate_csv(const std::string& filename, int num_rows) {
    // Open the file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Set up random number generation
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> int_distr(1, 10000); // Integer range
    std::uniform_real_distribution<> float_distr(1.0, 10000.0); // Float range

    // Write the header
    file << "ID,Col1,Col2,Col3,Col4\n";

    // Write the data
    for (int i = 0; i < num_rows; ++i) {
        file << int_distr(gen) << "," 
             << int_distr(gen) << "," 
             << int_distr(gen) << "," 
             << std::fixed << std::setprecision(5) << float_distr(gen) << "," 
             << std::fixed << std::setprecision(5) << float_distr(gen) << "\n";
    }

    // Close the file
    file.close();
    std::cout << "CSV file generated: " << filename << std::endl;
}

int main () {
    generate_csv("smallerrandnumbers.csv", 1000);
    return 0;
}